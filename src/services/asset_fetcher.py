import os
import base64
import re
import requests
from PIL import Image
from io import BytesIO
from ddgs import DDGS
from groq import Groq
from src.config.settings import TEMP_DIR, PEXELS_API_KEY, GROQ_API_KEY, GROQ_VISION_MODEL

HEADERS = {"User-Agent": "Mozilla/5.0"}
BLOCKED_DOMAINS = ["shutterstock", "gettyimages", "alamy", "istockphoto", "dreamstime"]
MAX_ASPECT_RATIO = 1.5
MAX_REFINE_ATTEMPTS = 4

groq_client = Groq(api_key=GROQ_API_KEY)


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

        response = groq_client.chat.completions.create(
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
        print(f"    [verify] Error: {e}")
        return (True, None)  # fallback: accept the image if verification fails


def _refine_search_term(original_term: str, line_text: str, image_expectation: str) -> str:
    """Ask Groq to refine the search term based on what the image expectation is."""
    try:
        response = groq_client.chat.completions.create(
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
    except Exception:
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


def _fetch_pexels(search_term: str, path: str) -> bool:
    if not PEXELS_API_KEY:
        return False
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": search_term, "per_page": 10, "orientation": "portrait"},
            timeout=8,
        )
        photos = r.json().get("photos", [])
        for photo in photos:
            url = photo.get("src", {}).get("large2x") or photo.get("src", {}).get("large")
            if url and _download_image(url, path, headers={"Authorization": PEXELS_API_KEY}):
                return True
    except Exception:
        pass
    return False


def _fetch_ddg(search_term: str, path: str) -> bool:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(search_term, max_results=15))
            for r in results:
                url = r.get("image", "")
                if any(s in url for s in BLOCKED_DOMAINS):
                    continue
                img_w = r.get("width", 1)
                img_h = r.get("height", 1)
                if img_h > 0 and (img_w / img_h) > MAX_ASPECT_RATIO:
                    continue
                if url and _download_image(url, path):
                    return True
    except Exception:
        pass
    return False


def _topic_keywords(topic: str) -> str:
    stopwords = {"what", "makes", "a", "an", "the", "and", "or", "why", "how",
                 "is", "are", "do", "does", "in", "on", "at", "to", "of", "for",
                 "your", "my", "our", "their", "its", "this", "that", "these", "those",
                 "always", "common", "look", "aren't", "isn't", "don't", "not",
                 "top", "5", "4", "3", "2", "1", "dark", "truth", "about", "human"}
    words = [w for w in topic.lower().split() if w.strip(".,?!'\"") not in stopwords]
    return " ".join(words[:4])


def fetch_assets(lines: list, topic: str = "") -> list:
    assets_dir = os.path.join(TEMP_DIR, "assets")
    keywords = _topic_keywords(topic) if topic else ""

    for line in lines:
        line_id = line["id"]
        original_search_term = line["search_term"]
        image_expectation = line.get("image_expectation", original_search_term)
        image_type = line.get("image_type", "stock")
        line_text = line.get("text", "")

        search_term = original_search_term
        if keywords and not any(k in search_term.lower() for k in keywords.split()):
            search_term = f"{keywords} {search_term}"

        path = os.path.join(assets_dir, f"{line_id}.jpg")

        print(f"  [asset] Line {line_id} [{image_type}]: '{search_term}'")

        success = False
        attempts = 0

        while attempts <= MAX_REFINE_ATTEMPTS and not success:
            # Try to fetch image
            if image_type == "stock":
                fetched = _fetch_pexels(search_term, path) or _fetch_ddg(search_term, path)
            else:
                fetched = _fetch_ddg(search_term, path) or _fetch_pexels(search_term, path)

            if fetched:
                # Verify with vision model
                print(f"    [verify] Checking if image matches...")
                is_valid, new_search_term = _verify_image(path, search_term, line_text, image_expectation)
                
                if is_valid:
                    print(f"  [asset] ✓ Verified")
                    success = True
                else:
                    # Get a better search term
                    if new_search_term and new_search_term != search_term:
                        print(f"    [verify] ✗ Image doesn't match, using new search term...")
                        search_term = new_search_term
                    else:
                        print(f"    [verify] ✗ Image doesn't match, refining search term...")
                        search_term = _refine_search_term(search_term, line_text, image_expectation)
                        print(f"    [verify] New term: '{search_term}'")
                    
                    attempts += 1
            else:
                # Try to refine search term if fetch failed
                search_term = _refine_search_term(search_term, line_text, image_expectation)
                print(f"    [verify] New term: '{search_term}'")
                attempts += 1

        if not success:
            print(f"  [asset] ✗ Failed for line {line_id}")
            path = None

        line["asset_path"] = path

    return lines
