"""Per-line query, license-safe image fetching (NO vision verification).

Strategy (validated on wet runs):
1. QUERY AGENT (local): the 10-lesson query agent (src/services/query_agent.py,
   now on Ollama) turns the whole script + topic into 5 precise search queries
   PER narration line (exact names, photo/era tokens, exclusions, a variety of
   visual angles) — never a single topic-level keyword list. No Groq/Gemini
   anywhere in this file.
2. Download candidate images per query from license-safe sources (Pexels,
   Wikimedia Commons, Openverse, DuckDuckGo restricted to Wikimedia-hosted
   files only), tagging each file with its origin line.
3. REJECT TEXT-SLOP deterministically (pytesseract OCR, no LLM): any image whose
   readable text covers > ASSET_MAX_TEXT_AREA of its area (captions/memes/big
   watermark blocks) is dropped.
4. NO PIXEL VERIFICATION: the per-line query engineering replaces vision.
   Facets/feature gates and model descriptions are gone from this path.
5. Assign kept images to narration lines origin-first (each line keeps the
   images its own queries found), with query/checklist text-overlap as tie-break
   and a local agentic plan (propose -> validate -> repair once) as the refined
   path; the deterministic greedy matcher is the fallback. Multi-image per line
   allowed; every line is guaranteed >= 1 image.
"""
import os
import re
import threading
import requests
import pytesseract
from PIL import Image
from io import BytesIO
from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config.settings import (
    TEMP_DIR,
    PEXELS_API_KEY,
    ASSET_MAX_PARALLEL_WORKERS,
    ASSET_MAX_ASPECT_RATIO,
    ASSET_TARGET_IMAGES,
    ASSET_MIN_IMAGES,
    ASSET_REJECT_TEXT_OVERLAY,
    ASSET_MAX_TEXT_AREA,
    ASSET_TEXT_MIN_CONF,
)
from src.services.asset_agent import assign as local_assign
from src.services.asset_agent import keywords as local_keywords
from src.services.query_agent import build_search_queries

HEADERS = {"User-Agent": "VideoPipeline/1.0 (history/mystery shorts; contact: local)"}
BLOCKED_DOMAINS = ["shutterstock", "gettyimages", "alamy", "istockphoto", "dreamstime"]
MAX_ASPECT_RATIO = ASSET_MAX_ASPECT_RATIO
MAX_PARALLEL_WORKERS = ASSET_MAX_PARALLEL_WORKERS
TARGET_IMAGES = ASSET_TARGET_IMAGES  # aim for this many topic images
MIN_IMAGES = ASSET_MIN_IMAGES        # hard stop: never ship a video with fewer
MAX_QUERY_SPECS = 18  # cap per-line queries fetched per round (round-robin)
REJECT_TEXT_OVERLAY = ASSET_REJECT_TEXT_OVERLAY  # OCR slop filter on/off
MAX_TEXT_AREA = ASSET_MAX_TEXT_AREA              # text coverage => image skipped
TEXT_MIN_CONF = ASSET_TEXT_MIN_CONF              # OCR word confidence floor

_file_hashes = {}
_file_hash_lock = threading.Lock()
_source_meta = {}   # path -> {"license": str, "source": str}


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


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    size = len(data)
    sample = data[:65536] + data[-65536:]
    return f"{size}:{hash(sample)}"


def _download_image(url: str, path: str, headers: dict = HEADERS) -> bool:
    try:
        r = requests.get(url, timeout=12, headers=headers)
        if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
            if _is_usable_image(r.content):
                with open(path, "wb") as f:
                    f.write(r.content)
                return True
    except Exception:
        pass
    return False


def _remember(path: str, license: str, source: str) -> bool:
    """Return True iff this is a NEW unique download. Records license+source."""
    h = _file_hash(path)
    with _file_hash_lock:
        if h in _file_hashes:
            return False
        _file_hashes[h] = path
    _source_meta.setdefault(path, {"license": license or "unspecified", "source": source or ""})
    return True


# ---------------------------------------------------------------------------
# COPYRIGHT-SAFE SOURCES
# ---------------------------------------------------------------------------
# Pexels: royalty-free, no attribution, commercial use. Always safe.
# Wikimedia Commons: explicit licenses; we keep only PD / CC0 / CC BY / CC BY-SA.
# Openverse: CC/PD index across many providers (Flickr, Wikimedia, Maxpixel...),
# returns the real license per image.
# DuckDuckGo: a general image source (not restricted to one host) — we set its
# license filter to Public/Share and block known watermarked stock aggregators,
# so only genuinely free photos get through. Never stops the whole source.

_SAFE_LICENSE_PATTERNS = [
    re.compile(r"^pd[\-a-z0-9]*$", re.I),               # pd, pd-old-70, pd-ineligible...
    re.compile(r"^public domain$", re.I),
    re.compile(r"^no restrictions$", re.I),
    re.compile(r"^cc\s*0$", re.I),                       # CC0
    re.compile(r"^cc\s*by(?![-a-z])($|$)", re.I),        # CC BY (no NC/ND suffix)
    re.compile(r"^cc\s*by\s*sa(?![-a-z])$", re.I),       # CC BY-SA
]


def _license_ok(license_str: str) -> bool:
    lic = (license_str or "").strip().lower()
    return any(p.search(lic) for p in _SAFE_LICENSE_PATTERNS)


def _fetch_pexels(search_term: str, base_path: str, count: int = 1) -> list:
    """Royalty-free images from Pexels. License: Pexels free license."""
    if not PEXELS_API_KEY:
        return []
    downloaded = []
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": search_term, "per_page": count * 3, "orientation": "portrait"},
            timeout=12,
        )
        photos = r.json().get("photos", [])
        for i, photo in enumerate(photos):
            if len(downloaded) >= count:
                break
            url = photo.get("src", {}).get("large2x") or photo.get("src", {}).get("large")
            if url:
                path = f"{base_path}_p{i}.jpg"
                if _download_image(url, path, headers={"Authorization": PEXELS_API_KEY}) \
                        and _remember(path, "Pexels free license", "Pexels"):
                    downloaded.append(path)
    except Exception as e:
        print(f"      [pexels] Error: {e}")
    return downloaded


def _fetch_wikimedia_commons(search_term: str, base_path: str, count: int = 1) -> list:
    """Wikimedia Commons images with explicit, allowed licenses (PD/CC0/CC BY/CC BY-SA)."""
    downloaded = []
    try:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {search_term}",
            "gsrnamespace": 6,
            "gsrlimit": count * 4,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size",
            "iiurlwidth": 1400,
        }
        r = requests.get("https://commons.wikimedia.org/w/api.php", params=params,
                         headers={"User-Agent": HEADERS["User-Agent"]}, timeout=15)
        pages = (r.json().get("query", {}) or {}).get("pages", {}) or {}
        for key in sorted(pages, key=lambda k: pages[k].get("index", 0)):
            if len(downloaded) >= count:
                break
            page = pages[key]
            ii = (page.get("imageinfo") or [{}])[0]
            url = ii.get("thumburl") or ii.get("url")
            lic = ""
            for field in ("LicenseShortName", "LicenseSpdx", "UsageTerms"):
                v = ((ii.get("extmetadata") or {}).get(field) or {}).get("value", "")
                if v:
                    lic = v
                    break
            if not url or not _license_ok(lic):
                continue
            w, h = ii.get("width", 1), ii.get("height", 1)
            try:
                if h and (w / h) > MAX_ASPECT_RATIO:
                    continue
            except (TypeError, ZeroDivisionError):
                pass
            path = f"{base_path}_c{len(downloaded)}.jpg"
            if _download_image(url, path) and _remember(path, lic, "Wikimedia Commons"):
                downloaded.append(path)
    except Exception as e:
        print(f"      [commons] Error: {e}")
    return downloaded


# Stock aggregators that always need paid licences / watermark their images.
_DDG_BLOCKED_HOSTS = BLOCKED_DOMAINS + [
    "alamy.com", "gettyimages.com", "shutterstock.com", "istockphoto.com",
    "dreamstime.com", "123rf.com", "adobestock.com", "depositphotos.com",
    "bigstockphoto.com", "corbisimages.com", "agefotostock.com", "graphicstock.com",
    "pinterest.com", "pinimg.com", "dailymail.co.uk", "dailymail.com",
    "dailystar.co.uk", "newscom.com", "flash89.com",
]

# A known-free fallback: if an image eventually has no license info, these
# hosts are trusted to be free/copyright-safe (Commons + Openverse CDNs).
_FREE_HOST_SUFFIXES = ("upload.wikimedia.org", "upload.wikimediausercontent.org",
                       "live.staticflickr.com", "openverse.org")


def _fetch_ddg(search_term: str, base_path: str, count: int = 1) -> list:
    """DuckDuckGo as a general source with license filter + stock blocklist.

    Not restricted to a single host. Uses DDG's `license:` filter for
    free-to-use photos and drops the known watermark/paid-stock hosts; images
    that slip through still pass the Groq feature check downstream.
    """
    downloaded = []
    try:
        with DDGS() as ddgs:
            for lic in ("Public", "Share"):
                if len(downloaded) >= count:
                    break
                try:
                    results = list(ddgs.images(
                        search_term, max_results=count * 5, license_image=lic))
                except Exception:
                    continue
                for i, r in enumerate(results):
                    if len(downloaded) >= count:
                        break
                    url = r.get("image", "")
                    if any(host in url.lower() for host in _DDG_BLOCKED_HOSTS):
                        continue
                    path = f"{base_path}_d{i}.jpg"
                    if url and _download_image(url, path) and \
                            _remember(path, "Free-to-use (search)", url[:120]):
                        downloaded.append(path)
    except Exception as e:
        print(f"      [ddg] Error: {e}")
    return downloaded


def _fetch_openverse(search_term: str, base_path: str, count: int = 1) -> list:
    """Openverse: CC/PD images from many providers with real license per result."""
    downloaded = []
    try:
        r = requests.get(
            "https://api.openverse.org/v1/images/",
            params={
                "q": search_term,
                "license": "by,by-sa,cc0,pdm",
                "license_type": "commercial",
                "page_size": count * 4,
            },
            headers={"User-Agent": HEADERS["User-Agent"]},
            timeout=15,
        )
        for it in r.json().get("results", []):
            if len(downloaded) >= count:
                break
            url = it.get("url") or ""
            w = it.get("width") or 0
            h = it.get("height") or 0
            if not url or (h and (w / h) > MAX_ASPECT_RATIO):
                continue
            lic = it.get("license") or "cc0"
            ver = it.get("license_version") or ""
            label = f"{lic} {ver}".strip()
            path = f"{base_path}_o{len(downloaded)}.jpg"
            if _download_image(url, path) and _remember(path, label, it.get("provider", "Openverse")):
                downloaded.append(path)
    except Exception as e:
        print(f"      [openverse] Error: {e}")
    return downloaded


# ---------------------------------------------------------------------------
# 1. KEYWORDS + TOPIC FACETS
# ---------------------------------------------------------------------------

_GENERIC = {"photo", "image", "picture", "people", "person", "view", "scene",
            "day", "night", "outdoor", "group", "crowd", "man", "woman", "building",
            "walking", "standing", "showing", "old", "big", "large", "two", "one"}


def _generate_topic_keywords(topic: str) -> list:
    """Local agent: topic -> 6 findable keyword phrases (heuristic fallback)."""
    return local_keywords.topic_keywords(topic)


# ---------------------------------------------------------------------------
# 2. DOWNLOAD
# ---------------------------------------------------------------------------

def _fetch_for_keyword(keyword: str, kw_idx: int, assets_dir: str, per_kw: int) -> list:
    base = os.path.join(assets_dir, f"kw{kw_idx}")
    paths = _fetch_pexels(keyword, base, per_kw)
    paths.extend(_fetch_wikimedia_commons(keyword, f"{base}c", per_kw))
    paths.extend(_fetch_openverse(keyword, f"{base}o", per_kw))
    paths.extend(_fetch_ddg(keyword, f"{base}d", per_kw))
    return paths


def _round_robin_specs(specs: list, max_specs: int) -> list:
    """Take at most max_specs (query, line_id) specs while keeping every line's
    queries interleaved, so no line starves on download budget."""
    if len(specs) <= max_specs:
        return specs
    queues = {}
    for q, lid in specs:
        queues.setdefault(lid, []).append(q)
    out, idx = [], 0
    while len(out) < max_specs:
        progressed = False
        for lid in queues:
            if idx < len(queues[lid]):
                out.append((queues[lid][idx], lid))
                progressed = True
                if len(out) >= max_specs:
                    return out
        if not progressed:
            break
        idx += 1
    return out


def _download_candidates(query_specs: list, assets_dir: str) -> dict:
    """Download candidates for each (query, line_id) spec in parallel.
    Returns {path: (query, line_id)} so assignment keeps the origin-line context."""
    per_q = max(1, int(round(TARGET_IMAGES * 1.6 / max(1, len(query_specs)))))
    candidates = {}
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(query_specs))) as ex:
        futs = {ex.submit(_fetch_for_keyword, q, i, assets_dir, per_q): (q, lid)
                for i, (q, lid) in enumerate(query_specs)}
        for fut in as_completed(futs):
            q, lid = futs[fut]
            try:
                for p in fut.result():
                    candidates[p] = (q, lid)
            except Exception as e:
                print(f"      [asset] query '{q}' error: {e}")
    return candidates


# ---------------------------------------------------------------------------
# 3. FILTER (deterministic OCR text-overlay rejection) — no vision
# ---------------------------------------------------------------------------

def _image_has_text_overlay(path: str) -> bool:
    """Deterministic OCR text-overlay check (pytesseract, no LLM).
    True if readable words cover > MAX_TEXT_AREA fraction of the image — catches
    photos ruined by captions/memes/big watermarks. Tiny credit marks pass."""
    if not REJECT_TEXT_OVERLAY:
        return False
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        area = float(w * h)
        if area <= 0:
            return False
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,
                                         config="--oem 3 --psm 11")
        covered = 0.0
        for conf, txt, l, t, bw, bh in zip(
                data["conf"], data["text"], data["left"], data["top"],
                data["width"], data["height"]):
            if txt and txt.strip() and float(conf) >= TEXT_MIN_CONF:
                covered += float(bw) * float(bh)
        return covered / area > MAX_TEXT_AREA
    except Exception as e:
        return False


def _describe_and_filter_candidates(candidates: dict, facets_tokens: set | None = None, topic: str = "") -> dict:
    """Deterministic filter only (OCR text-overlay rejection) — no vision.
    candidates: {path: (query, line_id)}. Returns {path: (query, query, line_id)}
    in insertion order; the query doubles as the image's feature text. Pixels are
    never described or verified: the per-line query agent pins images to topic."""
    """Deterministic filter only (OCR text-overlay rejection) — no vision.
    candidates: {path: (query, line_id)}. Returns {path: (query, query, line_id)}
    in insertion order; the query doubles as the image's feature text.
    The facets_tokens gate is intentionally removed: the per-line query agent
    already pins images to the topic, so pixel-level verification is gone."""
    kept = {}
    for p, (q, lid) in candidates.items():
        if _image_has_text_overlay(p):
            print(f"      [text] ✗ text overlay (> {MAX_TEXT_AREA:.0%} area) — skip: {os.path.basename(p)}")
            continue
        kept[p] = (q, q, lid)
    return kept


# ---------------------------------------------------------------------------
# 4. ASSIGN (query / origin + checklist overlap) — images -> narration lines
# ---------------------------------------------------------------------------

def _tokens(s: str):
    return set(re.findall(r"[a-z0-9]{3,}", s.lower())) - _GENERIC


def _greedy_assign(images: list, lines: list) -> dict:
    """images: [(path, feature_str, label, best_line_default)] -> {line_id: [paths]}."""
    mapping = {ln["id"]: [] for ln in lines}
    used = set()
    for ln in lines:
        want = _tokens(ln.get("image_features", "") or ln.get("image_expectation", "") or ln.get("search_term", ""))
        if not want:
            continue
        ranked = sorted(images, key=lambda im: -len(want & _tokens(im[1])))
        for im in ranked:
            if im[2] in used:
                continue
            if want & _tokens(im[1]):
                mapping[ln["id"]].append(im[2])
                used.add(im[2])
                break
    for im in images:
        if im[2] not in used and im[3] is not None:
            mapping[im[3]].append(im[2])
            used.add(im[2])
    return mapping


def _assign_images_to_lines(images: dict, lines: list, topic: str) -> dict:
    """images: {path: (features, query, line_id)}. Returns {line_id: [paths]}.
    Local agent proposes the plan (with one repair pass); the deterministic
    greedy token-overlap matcher fills in entirely when no usable plan came
    back. No Groq/Gemini is involved in assignment anymore."""
    if not images or not lines:
        return {ln["id"]: list(images.keys()) for ln in lines}

    mapping = local_assign.plan_assignment(images, lines, topic)

    if not any(v for v in mapping.values()):
        print("    [assign] Greedy feature-match fallback (origin-line default)")
        ids = {ln["id"] for ln in lines}
        imgs = []
        for p in images:
            best = None
            best_score = -1
            for ln in lines:
                want = _tokens(ln.get("image_features", "") or ln.get("image_expectation", "") or ln.get("search_term", ""))
                score = len(_tokens(images[p][0] or images[p][1]) & want)
                if score > best_score:
                    best_score = score
                    best = ln["id"]
            if best is None and images[p][2] in ids:
                best = images[p][2]
            imgs.append((p, images[p][0] or images[p][1], p, best))
        mapping = _greedy_assign(imgs, lines)

    # Spread unused images onto empty lines.
    assigned = {p for deck in mapping.values() for p in deck}
    unassigned = [p for p in images if p not in assigned]
    empty_lines = [ln for ln in lines if not mapping.get(ln["id"])]
    if unassigned and empty_lines:
        for img_path in unassigned:
            if not empty_lines:
                break
            ln = empty_lines.pop(0)
            mapping.setdefault(ln["id"], []).append(img_path)

    return mapping


# ---------------------------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------------------------

def fetch_assets(lines: list, topic: str = "") -> list:
    """Per-line query, OCR-clean, license-safe asset fetch (no vision)."""
    assets_dir = os.path.join(TEMP_DIR, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    print(f"  [asset] Topic: {topic}")
    print("  [asset] Query agent: one search-query set per narration line...")
    query_map = build_search_queries(lines, topic)

    specs, seen = [], set()
    for ln in lines:
        for q in ln.get("search_queries", []) or []:
            if q and q not in seen:
                seen.add(q)
                specs.append((q, ln["id"]))
    if len(specs) < 4:
        print("  [asset] Query agent thin - supplementing with topic keywords")
        for kw in _generate_topic_keywords(topic):
            if kw not in seen:
                seen.add(kw)
                specs.append((kw, None))
    specs = _round_robin_specs(specs, MAX_QUERY_SPECS)
    print(f"  [asset] {len(specs)} queries across {len(query_map)} line(s)")

    print(f"  [asset] Downloading candidates (Pexels/Commons/Openverse/DDG-commons)...")
    candidates = _download_candidates(specs, assets_dir)
    print(f"  [asset] Downloaded {len(candidates)} candidate(s)")

    kept = _describe_and_filter_candidates(candidates)
    print(f"  [asset] OCR-clean, kept: {len(kept)}")

    rounds = 0
    while len(kept) < MIN_IMAGES and rounds < 3:
        rounds += 1
        print(f"  [asset] Only {len(kept)} images — refetch round {rounds}...")
        extra = _download_candidates(specs, assets_dir)
        for p, v in _describe_and_filter_candidates(extra).items():
            if p not in kept:
                kept[p] = v
        print(f"  [asset] Kept now: {len(kept)}")

    if len(kept) < MIN_IMAGES:
        print(f"  [asset] ⚠️  Only {len(kept)} images (wanted >= {MIN_IMAGES})")

    print(f"  [asset] Assigning {len(kept)} image(s) to {len(lines)} line(s)...")
    mapping = _assign_images_to_lines(kept, lines, topic)

    used = set()
    for i, line in enumerate(lines):
        deck = [p for p in mapping.get(line["id"], []) if p in kept and os.path.exists(p)]
        line["asset_paths"] = deck
        line["asset_path"] = deck[0] if deck else None
        for p in deck:
            used.add(p)
        lic = _source_meta.get(deck[0], {}).get("license", "") if deck else ""
        if lic:
            line["image_license"] = lic
        print(f"  [asset] Line {line['id']}: {len(deck)} image(s) [{lic}]")

    # Safety net: no line ships without a visual — reuse a kept image.
    pool = list(kept.keys())
    if pool:
        cursor = 0
        for i, line in enumerate(lines):
            if not line.get("asset_paths"):
                prev = lines[i - 1].get("asset_paths") if i > 0 else None
                if prev:
                    pick = prev[0]
                else:
                    pick = pool[cursor % len(pool)]
                    cursor += 1
                line["asset_paths"] = [pick]
                line["asset_path"] = pick
                lic = _source_meta.get(pick, {}).get("license", "")
                if lic:
                    line["image_license"] = lic
                print(f"  [asset] Line {line['id']}: backfilled 1 image [{lic}]")

    print(f"  [asset] Done — {len(used)}/{len(kept)} images used across lines")
    return lines
