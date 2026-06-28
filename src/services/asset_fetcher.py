import os
import requests
from PIL import Image
from io import BytesIO
from ddgs import DDGS
from src.config.settings import TEMP_DIR, PEXELS_API_KEY

HEADERS = {"User-Agent": "Mozilla/5.0"}
BLOCKED_DOMAINS = ["shutterstock", "gettyimages", "alamy", "istockphoto", "dreamstime"]

# Reject images that are too wide relative to their height (bad for 9:16 video)
# e.g. ratio > 1.5 means width is 1.5x the height — panoramic/landscape junk
MAX_ASPECT_RATIO = 1.5


def _is_usable_image(content: bytes) -> bool:
    """Check image is large enough and not too wide."""
    try:
        img = Image.open(BytesIO(content))
        w, h = img.size
        if h == 0:
            return False
        ratio = w / h
        return ratio <= MAX_ASPECT_RATIO and len(content) > 5000
    except Exception:
        return False


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
    """Search Pexels for high-quality stock photos."""
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
    """Search the web via DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(search_term, max_results=15))
            for r in results:
                url = r.get("image", "")
                if any(s in url for s in BLOCKED_DOMAINS):
                    continue
                # Also use DDG's reported dimensions to pre-filter wide images
                img_w = r.get("width", 1)
                img_h = r.get("height", 1)
                if img_h > 0 and (img_w / img_h) > MAX_ASPECT_RATIO:
                    continue
                if url and _download_image(url, path):
                    return True
    except Exception:
        pass
    return False


def fetch_assets(lines: list, topic: str = "") -> list:
    assets_dir = os.path.join(TEMP_DIR, "assets")

    for line in lines:
        line_id = line["id"]
        search_term = line["search_term"]
        image_type = line.get("image_type", "stock")

        # Always anchor to the main topic
        if topic and topic.lower() not in search_term.lower():
            search_term = f"{topic} {search_term}"

        path = os.path.join(assets_dir, f"{line_id}.jpg")

        print(f"  [asset] Line {line_id} [{image_type}]: '{search_term}'")

        success = False

        if image_type == "stock":
            if _fetch_pexels(search_term, path):
                print(f"  [asset] ✓ Pexels")
                success = True
            elif _fetch_ddg(search_term, path):
                print(f"  [asset] ✓ DDG fallback")
                success = True
        else:
            if _fetch_ddg(search_term, path):
                print(f"  [asset] ✓ DDG")
                success = True
            elif _fetch_pexels(search_term, path):
                print(f"  [asset] ✓ Pexels fallback")
                success = True

        if not success:
            print(f"  [asset] ✗ Failed for line {line_id}")
            path = None

        line["asset_path"] = path

    return lines
