import os
import base64
import re
import requests
from PIL import Image
from io import BytesIO
from ddgs import DDGS
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config.settings import TEMP_DIR, PEXELS_API_KEY, GROQ_API_KEY, GROQ_API_KEY_BACKUP, GROQ_VISION_MODEL

HEADERS = {"User-Agent": "Mozilla/5.0"}
BLOCKED_DOMAINS = ["shutterstock", "gettyimages", "alamy", "istockphoto", "dreamstime"]
MAX_ASPECT_RATIO = 1.5
MAX_REFINE_ATTEMPTS = 4
MAX_PARALLEL_WORKERS = 8  # Conservative: safe for most PCs
IMAGES_PER_LINE = 3  # Number of images to fetch per sentence

# Copyright-safe image sources only
# Pexels: Free, no attribution required, commercial use OK
# DuckDuckGo with license filter: Public domain and Creative Commons

groq_client = Groq(api_key=GROQ_API_KEY)
groq_backup_client = None
using_backup = False

def _get_vision_client():
    """Get current vision client (primary or backup)."""
    global groq_client, groq_backup_client, using_backup
    
    if using_backup and groq_backup_client:
        return groq_backup_client
    return groq_client

def _switch_to_backup_vision():
    """Switch to backup API key for vision calls."""
    global groq_backup_client, using_backup
    
    if GROQ_API_KEY_BACKUP and not using_backup:
        print("    [vision] Switching to backup API key...")
        groq_backup_client = Groq(api_key=GROQ_API_KEY_BACKUP)
        using_backup = True
        return True
    return False


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


def _verify_image(image_path: str, search_term: str, line_text: str, image_expectation: str) -> tuple:
    """Use Groq vision to check if image matches what we want.
    Returns (is_valid, new_search_term or None).
    """
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        current_client = _get_vision_client()
        response = current_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""You are validating an image for a video script.

Search term used: {search_term}
Sentence context: {line_text}
Expected visual concept: {image_expectation}

Question: Does this image show what the expected visual concept describes?
Answer ONLY 'valid' or 'invalid'.
If invalid, provide a better search term (3-5 words) that would find an image matching the expected visual concept.

Format (MUST include both lines):
Answer: valid/invalid
Search term: <new_search_term>"""},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            temperature=0.2,
            max_tokens=50,
        )
        answer = response.choices[0].message.content.strip().lower()
        
        # Parse response
        is_valid = "valid" in answer or "invalid" not in answer
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
        error_msg = str(e).lower()
        print(f"    [verify] Error: {e}")
        
        # Handle rate limit for vision calls
        if "rate_limit" in error_msg and not using_backup:
            if _switch_to_backup_vision():
                print(f"    [verify] Retrying with backup vision key...")
                return _verify_image(image_path, search_term, line_text, image_expectation)
        
        return (True, None)  # fallback: accept the image if verification fails


def _refine_search_term(original_term: str, line_text: str, image_expectation: str) -> str:
    """Ask Groq to refine the search term based on what the image expectation is."""
    try:
        current_client = _get_vision_client()
        response = current_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": f"""Original search term: '{original_term}'
Sentence context: '{line_text}'
Expected visual concept: '{image_expectation}'

The original search term didn't find a good image. Provide a better, more specific search term that would capture the expected visual concept (just the term, no explanation):"""
            }],
            temperature=0.7,
            max_tokens=50,
        )
        refined = response.choices[0].message.content.strip()
        refined = _strip_thinking_tags(refined)
        return refined if refined else original_term
    except Exception as e:
        error_msg = str(e).lower()
        print(f"    [refine] Error: {e}")
        
        # Handle rate limit for refine calls
        if "rate_limit" in error_msg and not using_backup:
            if _switch_to_backup_vision():
                print(f"    [refine] Retrying with backup vision key...")
                return _refine_search_term(original_term, line_text, image_expectation)
        
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


def _fetch_single_asset(line: dict, keywords: str, assets_dir: str) -> dict:
    """Fetch multiple assets for a single line. Returns the line with asset_paths list."""
    line_id = line["id"]
    original_search_term = line["search_term"]
    image_expectation = line.get("image_expectation", original_search_term)
    image_type = line.get("image_type", "stock")
    line_text = line.get("text", "")

    search_term = original_search_term
    if keywords and not any(k in search_term.lower() for k in keywords.split()):
        search_term = f"{keywords} {search_term}"

    base_path = os.path.join(assets_dir, f"{line_id}.jpg")

    print(f"  [asset] Line {line_id} [{image_type}]: Fetching {IMAGES_PER_LINE} images for '{search_term}'")

    downloaded_paths = []
    attempts = 0

    while len(downloaded_paths) < IMAGES_PER_LINE and attempts <= MAX_REFINE_ATTEMPTS:
        # Calculate how many more images we need
        needed = IMAGES_PER_LINE - len(downloaded_paths)
        
        # Try to fetch images (Pexels first for stock, DDG first for search)
        if image_type == "stock":
            new_paths = _fetch_pexels(search_term, base_path, needed)
            if len(new_paths) < needed:
                # Try DDG as backup
                remaining = needed - len(new_paths)
                # Adjust base_path to avoid overwriting
                backup_base = os.path.join(assets_dir, f"{line_id}_ddg.jpg")
                new_paths.extend(_fetch_ddg(search_term, backup_base, remaining))
        else:
            new_paths = _fetch_ddg(search_term, base_path, needed)
            if len(new_paths) < needed:
                # Try Pexels as backup
                remaining = needed - len(new_paths)
                backup_base = os.path.join(assets_dir, f"{line_id}_pex.jpg")
                new_paths.extend(_fetch_pexels(search_term, backup_base, remaining))

        if new_paths:
            # Verify the first new image with vision model
            print(f"    [verify] Line {line_id}: Checking if images match...")
            is_valid, new_search_term = _verify_image(new_paths[0], search_term, line_text, image_expectation)
            
            if is_valid:
                print(f"  [asset] Line {line_id}: ✓ Downloaded {len(new_paths)} image(s)")
                downloaded_paths.extend(new_paths)
            else:
                # Images don't match, delete them and refine search
                for img_path in new_paths:
                    try:
                        os.remove(img_path)
                    except:
                        pass
                
                # Get a better search term
                if new_search_term and new_search_term != search_term:
                    print(f"    [verify] Line {line_id}: ✗ Images don't match, using new search term...")
                    search_term = new_search_term
                else:
                    print(f"    [verify] Line {line_id}: ✗ Images don't match, refining search term...")
                    search_term = _refine_search_term(search_term, line_text, image_expectation)
                    print(f"    [verify] Line {line_id}: New term: '{search_term}'")
                
                attempts += 1
        else:
            # No images fetched, refine search term
            search_term = _refine_search_term(search_term, line_text, image_expectation)
            print(f"    [verify] Line {line_id}: No images found, new term: '{search_term}'")
            attempts += 1

    if len(downloaded_paths) < IMAGES_PER_LINE:
        print(f"  [asset] Line {line_id}: ⚠️  Only got {len(downloaded_paths)}/{IMAGES_PER_LINE} images")
    
    if not downloaded_paths:
        print(f"  [asset] Line {line_id}: ✗ Failed to fetch any images")

    line["asset_paths"] = downloaded_paths  # Now a list instead of single path
    # Keep backward compatibility
    line["asset_path"] = downloaded_paths[0] if downloaded_paths else None
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
            executor.submit(_fetch_single_asset, line, keywords, assets_dir): line["id"]
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
