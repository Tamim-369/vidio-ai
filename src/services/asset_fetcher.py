import os
import base64
import re
import time
import threading
import requests
from PIL import Image
from io import BytesIO
from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config.settings import (
    TEMP_DIR,
    PEXELS_API_KEY,
    ASSET_MAX_PARALLEL_WORKERS,
    ASSET_IMAGES_PER_LINE,
    ASSET_MAX_REFINE_ATTEMPTS,
    ASSET_MAX_ASPECT_RATIO,
    ASSET_VERIFY_MIN_INTERVAL,
)
from src.services.llm import call_text

HEADERS = {"User-Agent": "Mozilla/5.0"}
BLOCKED_DOMAINS = ["shutterstock", "gettyimages", "alamy", "istockphoto", "dreamstime"]
MAX_ASPECT_RATIO = ASSET_MAX_ASPECT_RATIO
MAX_REFINE_ATTEMPTS = ASSET_MAX_REFINE_ATTEMPTS
MAX_PARALLEL_WORKERS = ASSET_MAX_PARALLEL_WORKERS  # Conservative: safe for most PCs
IMAGES_PER_LINE = ASSET_IMAGES_PER_LINE  # Number of images to fetch per sentence

# Rate-limit gate: Groq free-tier ITPM = 7000; each vision call ~2200 input tokens.
# Max safe throughput ≈ 3 calls/min → 19 s between calls.
_VERIFY_MIN_INTERVAL = ASSET_VERIFY_MIN_INTERVAL
_verify_lock = threading.Lock()
_verify_last_time = 0.0

def _rate_limit_verify():
    """Block until safe to make another vision API call."""
    global _verify_last_time
    with _verify_lock:
        now = time.time()
        wait = _VERIFY_MIN_INTERVAL - (now - _verify_last_time)
        if wait > 0:
            print(f"    [verify] Rate-limit gate: waiting {wait:.0f}s...", flush=True)
            time.sleep(wait)
        _verify_last_time = time.time()

# Copyright-safe image sources only
# Pexels: Free, no attribution required, commercial use OK
# DuckDuckGo with license filter: Public domain and Creative Commons


def _strip_thinking_tags(text: str) -> str:
    """Remove thinking tags like <think>...</think> from text."""
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def _is_usable_image(content: bytes) -> bool:
    try:
        img = Image.open(BytesIO(content))
        w, h = img.size
        if h == 0:
            return False
        ratio = w / h
        return ratio <= MAX_ASPECT_RATIO and len(content) > 5000
    except Exception:
        return False


def _verify_image(image_path: str, search_term: str, line_text: str, image_expectation: str, topic: str = "") -> tuple:
    """Use Groq vision to check if image matches what we want.
    Returns (is_valid, new_search_term or None).
    """
    _rate_limit_verify()
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        topic_block = f"\nVideo topic: {topic}" if topic else ""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"""You are validating a single image for a video script.{topic_block}

Search term used: {search_term}
Sentence context: {line_text}
Expected visual concept: {image_expectation}

Question: Does this image clearly and specifically show the expected VISUAL CONCEPT? A matching image shows the actual subject — not just a related or abstract scene.

Rules:
- 'valid' ONLY if the image genuinely and specifically depicts the expected subject{(' for THIS exact topic' if topic else '')}.
- 'invalid' if it is a generic/related scene, unrelated, or only vaguely connected.
- If invalid, give a better search term (3-5 words) describing a real, findable scene that WOULD match.

Format (MUST include both lines):
Answer: valid/invalid
Search term: <new_search_term>"""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }]
        answer = call_text(messages, temperature=0.0, max_tokens=400,
                           tag="verify", vision=True).lower()

        # Strict parsing: 'valid' only when the model explicitly says so.
        first_line = (answer.splitlines()[0] if answer else "").lower()
        is_valid = ("valid" in first_line) and ("invalid" not in first_line)
        new_search_term = None

        if not is_valid:
            # Try to extract new search term
            if "search term:" in answer:
                parts = answer.split("search term:", 1)
                if len(parts) > 1:
                    new_search_term = parts[1].strip().strip('"\'').split('\n')[0].strip()

                    new_search_term = _strip_thinking_tags(new_search_term)

        return (is_valid, new_search_term)

    except Exception as e:
        print(f"    [verify] Error: {e}")
        return (True, None)  # fallback: accept the image if verification fails


# ---- Batch verification + verdict cache ------------------------------------
# Verifying every candidate image in its own LLM round-trip was the dominant
# cost of the fetch phase (~95 serialized calls + rate-limit gates). Now all
# candidate images for a line are sent in ONE vision call, and verdicts are
# cached by content hash so a re-downloaded image is never re-verified.

_verdict_cache = {}
_verdict_lock = threading.Lock()


def _file_hash(path: str) -> str:
    """Cheap content hash (size + first/last 64KB) for dedup cache keys."""
    with open(path, "rb") as f:
        data = f.read()
    size = len(data)
    sample = data[:65536] + data[-65536:]
    return f"{size}:{hash(sample)}"


def _cached_verdict(path: str, expectation: str, topic: str) -> tuple or None:
    """Return (is_valid, new_search_term) from cache, or None to verify."""
    key = (_file_hash(path), expectation, topic)
    with _verdict_lock:
        return _verdict_cache.get(key)


def _store_verdict(path: str, expectation: str, topic: str, result: tuple) -> None:
    key = (_file_hash(path), expectation, topic)
    with _verdict_lock:
        _verdict_cache[key] = result


def _verify_images_batch(paths: list, search_term: str, line_text: str,
                         image_expectation: str, topic: str = "") -> dict:
    """Verify a batch of images in ONE vision call; returns {path: (valid, term)}.

    Each path gets its own "IMAGE n: valid/invalid" verdict line. Verdicts are
    cached by content hash so identical re-downloaded images skip re-verification.
    """
    results = {}

    # Resolve from cache first.
    pending = []
    for p in paths:
        cached = _cached_verdict(p, image_expectation, topic)
        if cached is not None:
            results[p] = cached
        else:
            pending.append(p)
    if pending:
        _rate_limit_verify()
    if not pending:
        return results

    try:
        topic_block = f"\nVideo topic: {topic}" if topic else ""
        parts = [{"type": "text", "text": f"""You are validating images for a video script.{topic_block}

Search term used: {search_term}
Sentence context: {line_text}
Expected visual concept: {image_expectation}

For EACH image below, decide whether it clearly and specifically shows the expected VISUAL CONCEPT. A matching image shows the actual subject — not just a related or abstract scene.

Rules:
- 'valid' ONLY if the image genuinely and specifically depicts the expected subject{(' for THIS exact topic' if topic else '')}.
- 'invalid' if it is a generic/related scene, unrelated, or only vaguely connected.

Format: answer exactly one line per image, in order:
IMAGE 1: valid/invalid
IMAGE 2: valid/invalid
...
No other text."""}]
        for path in pending:
            with open(path, "rb") as f:
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"}})

        answer = call_text([{"role": "user", "content": parts}], temperature=0.0,
                           max_tokens=300, tag="verify", vision=True).lower()

        # Parse "IMAGE n: valid/invalid" lines (accept "1: valid", "image 2 valid", etc.)
        verdicts = {}
        for m in re.finditer(r"image\s*(\d+)\s*[:.\-]?\s*(valid|invalid)", answer):
            verdicts[int(m.group(1)) - 1] = m.group(2) == "valid"
    except Exception as e:
        print(f"    [verify] Batch error: {e}")
        verdicts = {}  # fall back to accepting pending images (lenient on failure)

    for i, path in enumerate(pending):
        is_valid = verdicts.get(i, True)  # unparsed images accepted (lenient)
        results[path] = (is_valid, None)
        _store_verdict(path, image_expectation, topic, (is_valid, None))

    return results


def _refine_search_term(original_term: str, line_text: str, image_expectation: str, topic: str = "") -> str:
    """Ask an LLM to refine the search term based on what the image expectation is."""
    try:
        topic_line = f"Video topic: '{topic}'" if topic else ""
        messages = [{
            "role": "user",
            "content": f"""Original search term: '{original_term}'
Sentence context: '{line_text}'
Expected visual concept: '{image_expectation}'
{topic_line}

The original search term didn't find a good image. Provide a better, more specific search term that would find a REAL image of THIS topic (just the term, no explanation):"""
        }]
        refined = call_text(messages, temperature=0.7, max_tokens=200,
                            tag="refine", vision=True)
        refined = _strip_thinking_tags(refined)
        return refined if refined else original_term
    except Exception as e:
        print(f"    [refine] Error: {e}")
        return original_term


def _download_image(url: str, path: str, headers: dict = HEADERS) -> bool:
    try:
        r = requests.get(url, timeout=8, headers=headers)
        if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
            if _is_usable_image(r.content):
                with open(path, "wb") as f:
                    f.write(r.content)
                return True
    except Exception:
        pass
    return False


def _fetch_pexels(search_term: str, path: str, count: int = 1) -> list:
    """Fetch multiple images from Pexels (100% copyright-free, royalty-free).
    Returns list of successfully downloaded paths."""
    if not PEXELS_API_KEY:
        return []
    
    downloaded = []
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": search_term, "per_page": count * 3, "orientation": "portrait"},
            timeout=8,
        )
        photos = r.json().get("photos", [])
        
        for i, photo in enumerate(photos):
            if len(downloaded) >= count:
                break
                
            url = photo.get("src", {}).get("large2x") or photo.get("src", {}).get("large")
            if url:
                # Generate unique path for each image
                base, ext = os.path.splitext(path)
                img_path = f"{base}_{i+1}{ext}" if i > 0 else path
                
                if _download_image(url, img_path, headers={"Authorization": PEXELS_API_KEY}):
                    downloaded.append(img_path)
    except Exception as e:
        print(f"      [pexels] Error: {e}")
        pass
    
    return downloaded


def _fetch_ddg(search_term: str, path: str, count: int = 1) -> list:
    """Fetch multiple images from DuckDuckGo with license filtering.
    Returns list of successfully downloaded paths."""
    downloaded = []
    try:
        with DDGS() as ddgs:
            # Try with license filter first, fallback to regular if no results
            try:
                results = list(ddgs.images(
                    search_term, 
                    max_results=count * 5,
                    license_image='Public'  # Public domain + Creative Commons
                ))
            except:
                # Fallback: regular search but we'll filter manually
                results = list(ddgs.images(search_term, max_results=count * 5))
            
            for i, r in enumerate(results):
                if len(downloaded) >= count:
                    break
                    
                url = r.get("image", "")
                
                # Skip known copyright/watermarked sources
                if any(s in url.lower() for s in BLOCKED_DOMAINS):
                    continue
                
                # Skip obvious stock photo sites
                stock_indicators = ['stock', 'premium', 'watermark', 'preview']
                if any(indicator in url.lower() for indicator in stock_indicators):
                    continue
                    
                # Check aspect ratio
                img_w = r.get("width", 1) 
                img_h = r.get("height", 1)
                # Ensure both are numbers before comparison
                try:
                    img_w = float(img_w) if img_w else 1
                    img_h = float(img_h) if img_h else 1
                    if img_h > 0 and (img_w / img_h) > MAX_ASPECT_RATIO:
                        continue
                except (ValueError, TypeError):
                    # If width/height aren't valid numbers, skip aspect ratio check
                    pass
                
                # Generate unique path for each image
                base, ext = os.path.splitext(path)
                img_path = f"{base}_{i+1}{ext}" if i > 0 else path
                
                if url and _download_image(url, img_path):
                    downloaded.append(img_path)
    except Exception as e:
        print(f"      [ddg] Error: {e}")
        pass
    
    return downloaded


def _topic_keywords(topic: str) -> str:
    stopwords = {"what", "makes", "a", "an", "the", "and", "or", "why", "how",
                 "is", "are", "do", "does", "in", "on", "at", "to", "of", "for",
                 "your", "my", "our", "their", "its", "this", "that", "these", "those",
                 "always", "common", "look", "aren't", "isn't", "don't", "not",
                 "top", "5", "4", "3", "2", "1", "dark", "truth", "about", "human"}
    words = [w for w in topic.lower().split() if w.strip(".,?!'\"") not in stopwords]
    return " ".join(words[:4])


def _fetch_single_asset(line: dict, keywords: str, assets_dir: str, topic: str = "") -> dict:
    """Fetch multiple assets for a single line. Returns the line with asset_paths list."""
    line_id = line["id"]
    original_search_term = line["search_term"]
    image_expectation = line.get("image_expectation", original_search_term)
    image_type = line.get("image_type", "stock")
    line_text = line.get("text", "")

    search_term = original_search_term
    if keywords and not any(k in search_term.lower() for k in keywords.split()):
        search_term = f"{keywords} {search_term}"

    print(f"  [asset] Line {line_id} [{image_type}]: Fetching {IMAGES_PER_LINE} images for '{search_term}'")

    downloaded_paths = []
    attempts = 0
    rounds = 0
    tried_terms = set()  # search terms already fetched & verified (break circular loops)
    MAX_ROUNDS = MAX_REFINE_ATTEMPTS + IMAGES_PER_LINE * 2  # hard cap on fetch/verify rounds

    # IMPORTANT: every round must write to UNIQUE filenames. The Pexels/DDG
    # fetchers read a base path and derive `{base}_{i+1}.jpg` names, so we stamp
    # a per-round token into the base. If rounds reused the same base, a later
    # round would overwrite an already-validated file and its rejection could
    # os.remove() that file — leaving dead paths in downloaded_paths (the
    # "temp/assets/4_2.jpg" FileNotFound crash at assembly time).
    round_stamp = 0

    while len(downloaded_paths) < IMAGES_PER_LINE and attempts <= MAX_REFINE_ATTEMPTS and rounds < MAX_ROUNDS:
        rounds += 1
        round_stamp += 1
        # Calculate how many more images we need
        needed = IMAGES_PER_LINE - len(downloaded_paths)
        
        # Try to fetch images (Pexels first for stock, DDG first for search).
        # Unique base per round so no filename is ever written twice.
        if image_type == "stock":
            new_paths = _fetch_pexels(search_term, os.path.join(assets_dir, f"{line_id}_r{round_stamp}.jpg"), needed)
            if len(new_paths) < needed:
                # Try DDG as backup
                remaining = needed - len(new_paths)
                # Adjust base_path to avoid overwriting
                backup_base = os.path.join(assets_dir, f"{line_id}_r{round_stamp}_ddg.jpg")
                new_paths.extend(_fetch_ddg(search_term, backup_base, remaining))
        else:
            new_paths = _fetch_ddg(search_term, os.path.join(assets_dir, f"{line_id}_r{round_stamp}.jpg"), needed)
            if len(new_paths) < needed:
                # Try Pexels as backup
                remaining = needed - len(new_paths)
                backup_base = os.path.join(assets_dir, f"{line_id}_r{round_stamp}_pex.jpg")
                new_paths.extend(_fetch_pexels(search_term, backup_base, remaining))

        if not new_paths:
            # No images fetched, refine search term
            search_term = _refine_search_term(search_term, line_text, image_expectation)
            print(f"    [verify] Line {line_id}: No images found, new term: '{search_term}'")
            attempts += 1
            continue

        # Verify all candidates in ONE vision call (batch) and keep only matches.
        results = _verify_images_batch(new_paths, search_term, line_text, image_expectation, topic)

        gained = 0
        rejected = 0
        for img_path in new_paths:
            is_valid, _ = results.get(img_path, (False, None))
            if is_valid:
                downloaded_paths.append(img_path)
                gained += 1
            else:
                rejected += 1
                try:
                    os.remove(img_path)
                except OSError:
                    pass

        if gained:
            print(f"  [asset] Line {line_id}: ✓ Kept {gained} matching image(s) "
                  f"({len(downloaded_paths)}/{IMAGES_PER_LINE} so far)")
        else:
            print(f"    [verify] Line {line_id}: ✗ Nothing matched for '{search_term}'")
            if rejected:
                print(f"      [asset] Line {line_id}: {rejected} candidate(s) rejected by verifier")

        if gained and len(downloaded_paths) >= IMAGES_PER_LINE:
            break

        # Still need more images. Refine to a term we have NOT already fetched
        # and searched for — otherwise the same images come back and we loop.
        # Fall back to a "variation" of the last term only if the refiner keeps
        # returning something we've already tried.
        candidate = _refine_search_term(search_term, line_text, image_expectation, topic).strip()
        if not candidate or candidate.lower() == search_term.lower():
            # Keep moving: strip an adjective clause or nudge the phrasing so the
            # search actually differs from the attempt that just failed.
            candidate = f"{search_term} alternate view" if "alternate view" not in search_term else search_term + " ii"
        if candidate.lower() in tried_terms:
            candidate = f"{candidate} alternate view"
        search_term = candidate or search_term
        tried_terms.add(search_term.lower())
        attempts += 1

    if len(downloaded_paths) < IMAGES_PER_LINE and not downloaded_paths:
        print(f"  [asset] Line {line_id}: ✗ Failed to fetch any verified images")
    elif len(downloaded_paths) < IMAGES_PER_LINE:
        print(f"  [asset] Line {line_id}: ⚠️  Only kept {len(downloaded_paths)}/{IMAGES_PER_LINE} verified images")

    line["asset_paths"] = [p for p in downloaded_paths if os.path.exists(p)]
    # Keep backward compatibility
    line["asset_path"] = line["asset_paths"][0] if line["asset_paths"] else None
    return line


def fetch_assets(lines: list, topic: str = "") -> list:
    """Fetch assets for all lines in parallel using ThreadPoolExecutor."""
    assets_dir = os.path.join(TEMP_DIR, "assets")
    keywords = _topic_keywords(topic) if topic else ""

    print(f"  [asset] Starting parallel fetch with {MAX_PARALLEL_WORKERS} workers...")
    
    # Create a dictionary to maintain line order
    line_results = {}
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        # Submit all tasks
        future_to_line = {
            executor.submit(_fetch_single_asset, line, keywords, assets_dir, topic): line["id"]
            for line in lines
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_line):
            line_id = future_to_line[future]
            try:
                result_line = future.result()
                line_results[result_line["id"]] = result_line
            except Exception as e:
                print(f"  [asset] Line {line_id}: ✗ Error: {e}")
                # Find the original line and mark it as failed
                for line in lines:
                    if line["id"] == line_id:
                        line["asset_path"] = None
                        line_results[line_id] = line
                        break
    
    # Rebuild lines list in original order with updated asset paths
    for i, line in enumerate(lines):
        if line["id"] in line_results:
            lines[i] = line_results[line["id"]]
    
    print(f"  [asset] Parallel fetch complete!")
    return lines
