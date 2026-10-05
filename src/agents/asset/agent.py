"""Asset agent: per-line query, license-safe image fetching (NO vision).

Strategy (validated on wet runs):
1. QUERY AGENT: the 10-lesson query agent (src/agents/query/agent.py, on the
   central LLM seam) turns the whole script + topic into precise search queries
   PER narration line (exact names, photo/era tokens, exclusions, a variety of
   visual angles) — never a single topic-level keyword list. No Gemini
   anywhere in this file.
2. Download candidate images per query from license-safe sources (Pexels,
   Wikimedia Commons, Openverse, DuckDuckGo restricted to known-free hosts),
   tagging each file with its origin line. All providers refuse paid-stock
   and news-wire/press hosts (BLOCKED_DOMAINS + _BLOCKED_NEWS_SITES).
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
import time
import threading
from statistics import median
import requests
import pytesseract
from PIL import Image
from io import BytesIO
from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
ASSET_MAX_PARALLEL_WORKERS = int(os.getenv("ASSET_MAX_PARALLEL_WORKERS", "8"))
ASSET_MAX_ASPECT_RATIO = float(os.getenv("ASSET_MAX_ASPECT_RATIO", "1.5"))
# Topic-first asset strategy: per-line search queries from the query agent, keep
# at least MIN OCR-clean images, then assign them to lines (multi per line ok).
ASSET_TARGET_IMAGES = int(os.getenv("ASSET_TARGET_IMAGES", "6"))
ASSET_MIN_IMAGES = int(os.getenv("ASSET_MIN_IMAGES", "4"))
# Deterministic text-overlay slop filter (pytesseract OCR, no LLM): reject any
# image whose readable text covers more than ASSET_MAX_TEXT_AREA fraction of its
# area (photos with captions/memes/watermark blocks - not tiny credit marks).
ASSET_REJECT_TEXT_OVERLAY = os.getenv("ASSET_REJECT_TEXT_OVERLAY", "1") == "1"
ASSET_MAX_TEXT_AREA = float(os.getenv("ASSET_MAX_TEXT_AREA", "0.04"))
ASSET_TEXT_MIN_CONF = int(os.getenv("ASSET_TEXT_MIN_CONF", "50"))

from src.utils.file_helpers import workspace_path, topic_body_text
from urllib.parse import urlsplit

from src.agents.asset import assign as local_assign
from src.agents.asset import keywords as local_keywords
from src.agents.asset.commons import (
    ASSET_ENABLE_GENERIC as _ENABLE_GENERIC,
    ASSET_ENABLE_PEXELS as _ENABLE_PEXELS,
    COMMONS_PER_ENTITY,
    MAX_ENTITY_LEN,
)
from src.agents.asset.entities import extract_entities
from src.agents.asset.validate import (
    validate_candidate, is_non_visual_entity, resolve_hull, hull_designator,
)
from src.agents.query.agent import build_search_queries

HEADERS = {"User-Agent": "VideoPipeline/1.0 (history/mystery shorts; contact: local)"}
# Paid-stock watermarked hosts + news-wire/press hosts (their photos are
# copyrighted and cannot be used). Applied to EVERY provider, not just DDG.
# These are DISTINCTIVE substrings matched against the hostname only.
BLOCKED_DOMAINS = [
    # paid stock agencies with watermarks / licenses
    "shutterstock", "gettyimages", "alamy", "istockphoto", "dreamstime",
    "123rf", "adobestock", "depositphotos", "bigstockphoto", "corbisimages",
    "agefotostock", "graphicstock", "pinterest", "pinimg", "newscom",
    "flash89", "vecteezy", "envatostock", "storyblocks", "created-by",
    # news-wire / press image pools (copyrighted photojournalism)
    "reuters", "apimages", "apnews", "afpforum", "afp.com", "upi.com",
    "zumapress", "epaimages", "pressassociation", "pa.media", "getty.images",
    "wireimage", "newzones", "oddweirder", "imrg",
]
# Full site domains of major news outlets whose photos are all copyrighted.
_BLOCKED_NEWS_SITES = [
    "cnn.com", "foxnews.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com",
    "msnbc.com", "nytimes.com", "wsj.com", "washingtonpost.com",
    "theguardian.com", "dailymail.co.uk", "dailystar.co.uk", "express.co.uk",
    "thesun.co.uk", "mirror.co.uk", "telegraph.co.uk", "bbc.co.uk",
    "aljazeera.com", "upi.com", "reuters.com", "apnews.com",
]
# Openverse provider/source names that are news-wire pools (their CC/PD
# claim is unreliable for commercial use). Brand tokens only - matched
# case-insensitively against the provider/source field.
_NEWS_WIRE_PROVIDER_TOKENS = (
    "getty", "reuters", "associated press", "afp", "upi", "zuma press",
    "epaimages", "press association", "pa media", "ap images", "wire photos",
)
MAX_ASPECT_RATIO = ASSET_MAX_ASPECT_RATIO
MAX_PARALLEL_WORKERS = ASSET_MAX_PARALLEL_WORKERS
TARGET_IMAGES = ASSET_TARGET_IMAGES  # aim for this many topic images
MIN_IMAGES = ASSET_MIN_IMAGES        # hard stop: never ship a video with fewer
MAX_QUERY_SPECS = 18  # cap per-line queries fetched per round (round-robin)
REJECT_TEXT_OVERLAY = ASSET_REJECT_TEXT_OVERLAY  # OCR slop filter on/off
MAX_TEXT_AREA = ASSET_MAX_TEXT_AREA              # text coverage => image skipped
TEXT_MIN_CONF = ASSET_TEXT_MIN_CONF              # OCR word confidence floor

_file_hashes = {}       # (workspace, hash) -> path   - dedup scoped per video
_file_hash_lock = threading.Lock()
_source_meta = {}       # path -> {"license": str, "source": str}

# Wikimedia throttles burst API access with empty/non-JSON bodies, so the
# Commons *search* call is globally serialized with a small inter-call gap
# (Downloads to upload.wikimedia.org are separate hosts and stay parallel).
_COMMONS_SEARCH_LOCK = threading.Lock()
_commons_last = 0.0
_commons_gap_s = 0.35

# --- Era gate -------------------------------------------------------------
# A vision audit of a real pool (18 images, vision model ratings) found 9/18
# rated 0. Every one was MODERN: Soyuz/shuttle/Atlas launches returned by the
# query "Rocket" (6 images, 0 usable), plus a 1976 press photo in a 1942
# battle set. Grounding cannot catch this -- "V-2 rocket" really is in the
# article, so the term is valid; the archive's answer to it is modern.
#
# Commons already returns DateTime / ImageDescription / Categories in the
# extmetadata we were fetching and discarding, so the era signal is free and
# available BEFORE the image is downloaded. This gate drops the modern hits
# without spending a byte or a model call. Vision scoring then runs on
# survivors only.
#
# Slack is generous on purpose: it must not reject period-adjacent material
# (a 1970 photo of a 1943 U-boat is legitimate). It only rejects the wrong
# century.
ERA_SLACK_YEARS = int(os.getenv("ASSET_ERA_SLACK", "30"))

# Set once per video by _set_topic_era(). Empty = era unknown -> gate is a
# no-op, so a topic with no dates in it is never starved of images.
_topic_era_year = None

# Vehicle/program names that cannot appear in a pre-1950 subject. Only
# consulted when the topic itself is historical, so a modern-topic video
# (rockets, spaceflight) is never harmed.
_MODERN_MARKERS = (
    "soyuz", "space shuttle", "spaceshuttle", "atlas v", "atlas ii",
    "falcon 9", "falcon heavy", "starship", "spacex", "nasa",
    "progress msoyuz", "ariane", "space launch", "rocket launch",
    "saturday rocket", "long march", "vanguard", "ariane 5",
)

# Per-video metadata for downloaded assets, keyed by absolute path. Consumed by
# the vision gate later, so it is captured at download time.
_ASSET_META: dict = {}


def _set_topic_era(title: str, body: str) -> None:
    """Record the median year the topic is about, for the anachronism gate.

    Median, not max: an article about 1943 may mention 1945 and also 1918 in
    passing, and the central era is what we must not drift away from.
    """
    global _topic_era_year
    years = [int(y) for y in re.findall(r"\b(1[6-9]\d\d|20[0-2]\d)\b", f"{title} {body}")]
    years = [y for y in years if 1600 <= y <= 2025]
    _topic_era_year = int(median(years)) if years else None


def _meta_field(ii: dict, name: str) -> str:
    v = ((ii.get("extmetadata") or {}).get(name) or {}).get("value", "")
    if not isinstance(v, str):
        v = str(v)
    return re.sub(r"<[^>]+>", " ", v)


def _anachronistic(ii: dict, title: str) -> bool:
    """True when the file's own metadata places it far outside the topic era.

    Only fires when we know the topic's year; an unknown era passes everything
    through rather than silently emptying the pool.
    """
    if _topic_era_year is None:
        return False
    horizon = _topic_era_year + ERA_SLACK_YEARS

    # DateTimeOriginal, NOT DateTime. DateTime is the UPLOAD date: a 1942
    # photo of the Akagi sits on Commons with DateTime "2025-01-20". Using it
    # rejected essentially the entire archive (measured: 100% of a real pool
    # blocked, good images included) because archival uploads are mostly
    # recent. DateTimeOriginal is the date the work was made, and is populated
    # for the archival material this channel depends on.
    #   "19 May 1942" / "1942-06-06" / "4 June 1942 date QS:P571..."
    dt_orig = _meta_field(ii, "DateTimeOriginal")
    if dt_orig:
        years = re.findall(r"\b(1[6-9]\d\d|20\d\d)\b", dt_orig)
        if years and max(int(y) for y in years) > horizon:
            return True

    # Categories carry era info even when DateTime is missing or wrong, e.g.
    # "Space Shuttle launches" or "1960s photographs".
    cats = _meta_field(ii, "Categories")
    if cats:
        cat_years = re.findall(r"\b(1[6-9]\d\d|20\d\d)s?\b", cats)
        if cat_years and max(int(y.rstrip("s")) for y in cat_years) > horizon:
            return True
        low = cats.lower()
        if _topic_era_year < 1950 and any(m in low for m in _MODERN_MARKERS):
            return True

    # Last resort: the description naming a modern vehicle.
    desc = f"{_meta_field(ii, 'ImageDescription')} {title}".lower()
    if _topic_era_year < 1950 and any(m in desc for m in _MODERN_MARKERS):
        return True
    return False
# DuckDuckGo similarly rate-limits aggressive parallel scraping.
_DDG_SEM = threading.BoundedSemaphore(3)


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


def _is_blocked_host(url: str) -> bool:
    """True if the URL's host is a paid-stock or news-wire/press host."""
    try:
        host = (urlsplit(url).netloc or "").lower()
    except Exception:
        host = (url or "").lower()
    if not host:
        return False
    if any(site in host for site in _BLOCKED_NEWS_SITES):
        return True
    return any(token in host for token in BLOCKED_DOMAINS)


def _download_image(url: str, path: str, headers: dict = HEADERS) -> bool:
    if not url or _is_blocked_host(url):
        return False
    # One retry on a reset/timeout: Commons CDN resets connections under
    # parallel load, and a single dropped fetch used to shrink the pool.
    for attempt in range(2):
        try:
            r = requests.get(url, timeout=8, headers=headers)
            if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                if _is_usable_image(r.content):
                    with open(path, "wb") as f:
                        f.write(r.content)
                    return True
            return False
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 0:
                time.sleep(0.6)
                continue
        except Exception:
            return False
    return False


def _workspace_of(path: str) -> str:
    """The temp/<slug> root a path lives under (dedup scope for concurrency)."""
    parts = path.split(os.sep)
    if len(parts) >= 2 and parts[0] == "temp":
        return parts[1]
    return ""


def _remember(path: str, license: str, source: str) -> bool:
    """Return True iff this is a NEW unique download. Records license+source.
    Dedup is scoped per workspace so parallel videos never reject each other's
    (hash-identical) images, while within one video duplicates still collapse."""
    h = _file_hash(path)
    ws = _workspace_of(path)
    with _file_hash_lock:
        if (ws, h) in _file_hashes:
            return False
        _file_hashes[(ws, h)] = path
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


def _commons_search(params: dict) -> dict:
    """Run a Wikimedia Commons API search with the global burst throttle.

    Returns the parsed JSON dict ({} on failure). Wikimedia rate-limits burst
    API access with empty bodies AND HTML throttle pages, so a non-JSON body is
    treated as "no results" for the retry logic below.

    A ConnectionError (reset by peer) is retried with backoff: during a live
    pool run both Commons and Openverse started resetting connections
    mid-run, which silently cost us whole entities (Akagi and Kaga returned
    zero images while working minutes earlier).
    """
    with _COMMONS_SEARCH_LOCK:
        global _commons_last
        wait = _commons_gap_s - (time.monotonic() - _commons_last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get("https://commons.wikimedia.org/w/api.php", params=params,
                             headers={"User-Agent": HEADERS["User-Agent"]}, timeout=15)
        except (requests.ConnectionError, requests.Timeout):
            for attempt in range(2):
                time.sleep(1.5 * (attempt + 1))
                try:
                    r = requests.get("https://commons.wikimedia.org/w/api.php", params=params,
                                     headers={"User-Agent": HEADERS["User-Agent"]}, timeout=15)
                    break
                except (requests.ConnectionError, requests.Timeout):
                    continue
            else:
                return {}
        _commons_last = time.monotonic()
    if "json" not in r.headers.get("Content-Type", ""):
        return {}
    try:
        data = r.json()
    except ValueError:
        return {}
    return data or {}


def _fetch_wikimedia_commons(search_term: str, base_path: str, count: int = 1) -> list:
    """Wikimedia Commons images with explicit, allowed licenses (PD/CC0/CC BY/CC BY-SA).

    Runs at most twice: once for the term as written, and -- only if the term
    carries a hull designation that Commons contradicts -- once for the
    designation Commons actually agrees with.
    """
    # Correct the hull designation BEFORE downloading: the budget gets spent on
    # the first pass, so a post-download audit would be too late to matter.
    return _commons_pass(_audited_term(search_term), base_path, count)


def _audited_term(search_term: str) -> str:
    """Return the term with a hull number Commons agrees with, else unchanged.

    The audit is a metadata-only search (no image bytes) and is skipped for
    terms with no hull designation, so the common path costs nothing extra.
    """
    if not hull_designator(search_term):
        return search_term
    try:
        data = _commons_search(_commons_params(search_term, 6))
        pages = (data.get("query", {}) or {}).get("pages", {}) or {}
        cats = [_meta_field((pg.get("imageinfo") or [{}])[0], "Categories")
                for pg in pages.values()]
        corrected = resolve_hull(search_term, cats)
    except Exception:
        return search_term                      # never fail a term on an audit
    if corrected != search_term:
        print(f"      [entity] {search_term} -> {corrected} "
              f"(hull corrected: Commons categories never support the query)")
    return corrected


def _commons_params(search_term: str, count: int) -> dict:
    return {
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


def _commons_pass(search_term: str, base_path: str, count: int) -> list:
    """One Commons search+download pass."""
    downloaded = []
    try:
        params = _commons_params(search_term, count)
        data = _commons_search(params)
        pages = (data.get("query", {}) or {}).get("pages", {}) or {}
        if not pages:
            # A throttled burst returns an empty body; one quieter retry.
            data = _commons_search(params)
            pages = (data.get("query", {}) or {}).get("pages", {}) or {}
        for key in sorted(pages, key=lambda k: pages[k].get("index", 0)):
            page = pages[key]
            ii = (page.get("imageinfo") or [{}])[0]
            if len(downloaded) >= count:
                continue
            if not ii:
                continue
            url = ii.get("thumburl") or ii.get("url")
            lic = ""
            for field in ("LicenseShortName", "LicenseSpdx", "UsageTerms"):
                v = ((ii.get("extmetadata") or {}).get(field) or {}).get("value", "")
                if v:
                    lic = v
                    break
            if not url or not _license_ok(lic):
                continue
            # NOTE: an era gate on Commons metadata was tried here and
            # removed. Measured on a real pool it blocked 6/6 images a vision
            # model rated 5 (correct subject) while catching modern junk,
            # because both DateTime and DateTimeOriginal hold UPLOAD dates for
            # most archival files. It emptied the pool. Kept as _anachronistic
            # with tests documenting the failure. The real era + type check is
            # validate.py, which scores several metadata signals and only
            # rejects when they agree.
            w, h = ii.get("width", 1), ii.get("height", 1)
            try:
                if h and (w / h) > MAX_ASPECT_RATIO:
                    continue
            except (TypeError, ZeroDivisionError):
                pass

            # Per-image validation, from metadata only: no pixels are fetched
            # to decide, so this costs nothing. Catches maps, buildings,
            # memorials and logos returned for terms that name a region or a
            # command rather than an object.
            chk = validate_candidate(ii, page.get("title", ""), search_term,
                                     _topic_era_year)
            if chk["verdict"] == "reject":
                print(f"      [validate] ✗ {page.get('title','')[:44]} — "
                      f"{'; '.join(chk['reasons'][:2])}")
                continue
            path = f"{base_path}_c{len(downloaded)}.jpg"
            if _download_image(url, path) and _remember(path, lic, "Wikimedia Commons"):
                downloaded.append(path)
    except Exception as e:
        print(f"      [commons] Error: {e}")
    return downloaded


# DuckDuckGo can return arbitrary hosts, so it is restricted to a known-free
# whitelist (see _on_free_host). The global download gate
# (BLOCKED_DOMAINS + _BLOCKED_NEWS_SITES) additionally rejects paid-stock and
# news-wire hosts from EVERY provider.
# A known-free whitelist: DDG results are restricted to these hosts only
# (Commons + Flickr CC + Openverse CDN + a couple of trusted archives), so a
# random news or stock image host can never slip into the channel from DDG.
_FREE_HOST_SUFFIXES = ("upload.wikimedia.org", "upload.wikimediausercontent.org",
                       "live.staticflickr.com", "openverse.org",
                       "archive.org", "loc.gov", "nara.gov", "si.edu")


def _on_free_host(url: str) -> bool:
    try:
        host = (urlsplit(url).netloc or "").lower()
    except Exception:
        host = (url or "").lower()
    # Match the host EXACTLY or as a subdomain of an allowed suffix. A plain
    # substring test would let "loc.gov.attacker.com" (and lookalikes like
    # "xloc.gov") through the whitelist.
    host = host.split("@")[-1].split(":")[0]
    return any(host == sfx or host.endswith("." + sfx) for sfx in _FREE_HOST_SUFFIXES)


# Words too common to prove relevance on their own.
_STOP = {"a", "an", "the", "of", "and", "in", "on", "at", "to", "for", "with",
         "from", "by", "is", "was", "are", "were", "be", "as", "or", "it",
         "his", "her", "its", "their", "new", "old", "1", "2", "3"}


def _relevance_ok(text: str, search_term: str) -> bool:
    """True when a generic provider's alt text shares a content word with the
    query. DDG answered "USS Akagi Battle of Midway" with a stock photo of
    ducks in front of a flag, because nothing was checking what came back.
    Cheap and imperfect, but it stops the obvious non-sequiturs.
    """
    got = {w for w in _tokens(text) if w not in _STOP}
    want = {w for w in _tokens(search_term) if w not in _STOP}
    return bool(got & want)


def _fetch_ddg(search_term: str, base_path: str, count: int = 1) -> list:
    """DuckDuckGo as a fallback general source - FREE HOSTS ONLY.

    DDG results are restricted to hosts whose imagery is reliably free and
    stable over time (see _on_free_host whitelist), with DDG's `license:`
    filter on Public/Share as a second check. News-wire and paid-stock hosts
    cannot appear, so this source is copyright-safe.
    """
    downloaded = []
    try:
        with DDGS() as ddgs:
            for lic in ("Public", "Share"):
                if len(downloaded) >= count:
                    break
                try:
                    with _DDG_SEM:
                        results = list(ddgs.images(
                            search_term, max_results=count * 5, license_image=lic))
                except Exception:
                    continue
                for i, r in enumerate(results):
                    if len(downloaded) >= count:
                        break
                    url = r.get("image", "")
                    if not url or not _on_free_host(url):
                        continue
                    if not _relevance_ok(f"{r.get('title','')} "
                                          f"{r.get('image','')}", search_term):
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
            prov = it.get("provider") or it.get("source") or ""
            if any(tok in str(prov).lower() for tok in _NEWS_WIRE_PROVIDER_TOKENS):
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
    """Fetch images for one term across providers.

    Commons is the primary source and gets the whole budget. The generic
    web-image fallbacks are OFF by default: a Battle of Midway render came out
    with a photograph of ducks sitting in front of a flag, because DDG scrapes
    stock-photo sites and answers a warship query with whatever stock photo is
    nearest the keywords. Neither fallback carries the Commons extmetadata that
    validate_candidate() scores, so nothing screened them. Commons alone is
    both cleaner and verifiable, and it is not short of images for a historical
    topic.

    Set ASSET_ENABLE_GENERIC=1 to restore them; each result must then share a
    content word with the query or it is dropped.
    """
    base = os.path.join(assets_dir, f"kw{kw_idx}")
    paths = []
    if _ENABLE_PEXELS:
        paths = _fetch_pexels(keyword, base, per_kw)
    paths.extend(_fetch_wikimedia_commons(keyword, f"{base}c", per_kw))
    if _ENABLE_GENERIC:
        side = max(1, per_kw // 2)
        paths.extend(_fetch_openverse(keyword, f"{base}o", side))
        paths.extend(_fetch_ddg(keyword, f"{base}d", side))
    return paths


# Generic English nouns that are also common Commons search terms. Measured on
# a real pool: the term "Rocket" returned 6 images (Soyuz, shuttle, Atlas) and
# ZERO were usable, while "Schnorchel" returned 4 and ALL 4 were usable. The
# term is the whole story, so a generic term gets a token budget instead of a
# full slot. Cheap, instant, no model.
GENERIC_TERMS = {
    "rocket", "rockets", "ship", "ships", "plane", "planes", "war", "battle",
    "navy", "army", "tank", "tanks", "gun", "guns", "flag", "flags", "map",
    "maps", "train", "trains", "city", "island", "sea", "harbour", "harbor",
    "attack", "battlefield", "explosion", "fire", "soldier", "soldiers",
    "general", "admiral", "captain", "warship", "submarine", "submarines",
    "battleship", "carrier", "aircraft", "engine", "engines", "missile",
    "missiles", "torpedo", "torpedoes", "weapons", "military", "naval",
    "operation", "operations", "trial", "trials", "experiments", "research",
    "history", "museum", "memorial", "monument", "portrait", "painting",
}


def _entity_budget(entity: str) -> int:
    """How many images a term may contribute.

    A rare, multi-word, or ship-named term earns the full budget. A single
    common noun earns 1, because the audit showed a generic term spends the
    entire image allowance on unusable results and starves the good terms.
    """
    words = entity.split()
    if len(words) >= 2:
        return COMMONS_PER_ENTITY
    return 1 if entity.lower().strip(".,") in GENERIC_TERMS else COMMONS_PER_ENTITY


# Topic-anchored query state, set once per run by the pipeline.
_topic_anchor = [""]


def _event_anchor(topic: str) -> str:
    """The event words a search must carry to stay on-topic.

    A bare entity name is not a focused query. "USS Akagi" returns generic
    warship photography -- the fleet pictures that made a Midway video look
    like it was about "random boats". Pairing the entity with the event
    ("USS Akagi Battle of Midway") is what pulls battle-specific frames out of
    Commons instead of a generic carrier portrait.

    The topic's own headline is the cheapest reliable event name:
    "The Battle of Midway: How the US Navy ... 1942" -> "Battle of Midway".
    """
    head = (topic or "").split(":")[0]
    head = re.sub(r"^(the|a|an)\s+", "", head.strip(), flags=re.I)
    # Drop a leading year and trailing parenthetical year; keep the nouns.
    head = re.sub(r"\b(19|20)\d{2}\b", "", head)
    return re.sub(r"\s+", " ", head).strip(" -?:,")[:60]


def _focused_query(entity: str, topic: str) -> str:
    """Entity + event anchor, without duplicating words the entity already has.

    "USS Akagi" -> "USS Akagi Battle of Midway".
    "Battle of Midway" -> "Battle of Midway" (already focused).
    """
    anchor = _event_anchor(topic)
    if not anchor or not entity:
        return entity
    ent_tokens = {w.lower().strip(".,") for w in entity.split()}
    if {w.lower().strip(".,") for w in anchor.split()} <= ent_tokens:
        return entity                     # entity already names the event
    return f"{entity} {anchor}".strip()[:MAX_ENTITY_LEN]


def _entity_specs(entities: list, assets_dir: str) -> list:
    """Entity pool -> download specs. line_id is None: the pool is topic-level,
    so no image is 'owned' by a line until the relevance gate picks one.

    The 4th element is the per-term image budget (see _entity_budget)."""
    specs = []
    for i, e in enumerate(entities):
        e = _focused_query((e or "").strip(), _topic_anchor[0])
        if e:
            specs.append((e, None, os.path.join(assets_dir, f"ent{i}"),
                          _entity_budget(e)))
    return specs


def _download_entities(entities: list, assets_dir: str,
                       pool_cap: int = 0) -> dict:
    """Download a candidate pool from the topic entity pool.

    Returns {path: (entity, None)}. Commons is serialized by its own global
    throttle inside _commons_search; downloads stay parallel."""
    pool_cap = pool_cap or _pool_target(len(entities) and 10)
    specs = _entity_specs(entities, assets_dir)
    candidates = {}
    if not specs:
        return candidates

    # Bound total work BEFORE fetching: the pool is capped at MAX_POOL_IMAGES,
    # so budgets shrink as the entity count grows. Commons search is globally
    # throttled, so cost is roughly proportional to entity count; fetching
    # 14 entities x 6 = 84 searches took 6.4 minutes for a pool that ended up
    # 94 images deep and only 7 usable.
    n = len(specs)
    even = max(1, pool_cap // n)
    trimmed = [(e, base, min(b, even)) for (e, _l, base, b) in specs]
    if even < COMMONS_PER_ENTITY:
        print(f"  [asset] Pool capped: {n} entities x {even} images "
              f"(= {n * even} of max {pool_cap})")

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(trimmed))) as ex:
        futs = {ex.submit(_fetch_for_keyword, e, i, assets_dir, budget): e
                for i, (e, base, budget) in enumerate(trimmed)}
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                for p in fut.result():
                    candidates.setdefault(p, (e, None))
            except Exception as ex:
                print(f"      [asset] entity '{e}' error: {ex}")
    if len(candidates) > pool_cap:
        print(f"  [asset] Trimming pool {len(candidates)} -> {pool_cap}")
        candidates = dict(_diverse_slice(candidates, pool_cap))
    return candidates


def _diverse_slice(candidates: dict, limit: int) -> list:
    """Pick `limit` candidates round-robin across queries.

    The previous trim was "take the first N in arrival order", so whichever
    query happened to finish first took every slot -- measured 3 of 5 images
    from one query, all near-identical carrier portraits. Interleaving by query
    makes the pool diverse by construction, which is what assignment needs in
    order to tell images apart at all.
    """
    buckets: dict = {}
    for p, meta in candidates.items():
        buckets.setdefault((meta[1] or meta[0] or "").strip(), []).append(p)
    order = list(buckets)
    out: list = []
    while len(out) < limit and any(buckets.values()):
        for q in order:
            if buckets[q] and len(out) < limit:
                p = buckets[q].pop(0)
                out.append((p, candidates[p]))
    return out


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
    except Exception:
        return False


def _describe_and_filter_candidates(candidates: dict, facets_tokens: set | None = None, topic: str = "") -> dict:
    """Deterministic filter only (OCR text-overlay rejection) — no vision.
    candidates: {path: (query, line_id)}. Returns {path: (query, query, line_id)}
    in insertion order; the query doubles as the image's feature text.
    The facets_tokens gate is intentionally removed: the per-line query agent
    already pins images to the topic, so pixel-level verification is gone."""
    items = list(candidates.items())

    def _filter_pair(pair):
        p, (q, lid) = pair
        if _image_has_text_overlay(p):
            print(f"      [text] ✗ text overlay (> {MAX_TEXT_AREA:.0%} area) — skip: {os.path.basename(p)}")
            return None
        return p, (q, q, lid)

    if len(items) <= 1:
        results = [_filter_pair(pair) for pair in items]
    else:
        # pytesseract releases the GIL, so a thread pool scales the OCR pass.
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(items))) as ex:
            results = list(ex.map(_filter_pair, items))
    kept = {}
    for res in results:
        if res is not None:
            kept[res[0]] = res[1]
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
    back. No cloud LLM is involved in assignment anymore."""
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

    # Cap first, then fill. Capping drops over-represented terms, which can leave
    # lines empty, so the fill must run afterwards or the cap silently undoes
    # itself by starving a line.
    mapping = _cap_per_term(mapping, images, lines)

    # Distribution. The pool is small and focused (5 images for a whole video),
    # so it is SPREAD over the lines rather than spread thin: every image goes
    # to the line it matches best, a line may hold more than one, reuse across
    # lines is normal, and a line with no confident match is left bare.
    #
    # The previous contract was the opposite: "every line gets a distinct image"
    # plus a backfill loop that never left a line empty. With a 24-image pool
    # that forced a marginal photo onto every sentence, which is where the
    # "random boats" came from -- filler images, not best-match images. A
    # viewer cannot tell which of six generic carrier portraits is meant to be
    # Akagi, but a line that simply shows the one good Akagi photo reads fine.
    order = [ln["id"] for ln in lines]
    want = {ln["id"]: _tokens(ln.get("image_features", "")
                               or ln.get("image_expectation", "")
                               or ln.get("text", "")
                               or ln.get("search_term", ""))
            for ln in lines}

    # Rebuild the plan deterministically: score every (image, line) pair, then
    # give each line its best images up to the per-line cap.
    scores = []
    for p in images:
        feats = _tokens(images[p][0] or images[p][1] or "")
        for lid in order:
            scores.append((len(feats & want[lid]), lid, p))
    scores.sort(key=lambda t: (-t[0], t[1], t[2]))
    placed: set = set()

    out: dict = {lid: [] for lid in order}

    # Phase 1: give every image its single best-matching line, no repeats. This
    # is what stops the pool collapsing -- scoring every (image, line) pair and
    # filling to the cap sent the same best image to nine lines and left 3 of 5
    # images unused in the render.
    for score, lid, p in scores:
        if score <= 0:
            continue
        if p in placed:
            continue
        if len(out[lid]) >= MAX_IMAGES_PER_LINE:
            continue
        out[lid].append(p)
        placed.add(p)

    # Phase 2: reuse. A line under its cap takes another image only if one
    # genuinely matches it; otherwise it stays bare. Consecutive lines are
    # kept off the same picture -- "the same 2 images repeated over and over"
    # was the visible symptom of a pool with too little variety to alternate.
    for score, lid, p in scores:
        if score <= 0 or p in out[lid]:
            continue
        if len(out[lid]) >= MAX_IMAGES_PER_LINE:
            continue
        idx = order.index(lid)
        if idx and out.get(order[idx - 1]) and out[order[idx - 1]][-1] == p:
            continue                      # same image as the previous line
        out[lid].append(p)

    bare = [lid for lid in order if not out[lid]]
    if bare:
        print(f"    [assign] {len(bare)}/{len(order)} line(s) left with no image "
              f"(focused pool of {len(images)}, no matching query) - by design")
    return out


# A single search term may fill at most this many lines in one video. See
# _cap_per_term: "Midway" means the atoll, so Commons returned seabirds, and
# 6 of 6 assigned slots went to that one term.
MAX_IMAGES_PER_TERM = 2

# Pool size is bounded by the video, not by how many entities we found. A
# 10-line video needs ~2 images per line with room for assignment to reject
# some. Fetching 14 entities x 6 images cost 6.4 minutes of Commons throttling
# for a pool that assignment then used 7 images out of; and 94 images for a
# 10-line video is wasted download time, not more choice.
# The pool is deliberately SMALL. A 24-image pool for a 10-line video meant 2.4
# marginal photos per sentence, and marginal photos of warships look like
# random boats. Four to five focused images, spread over the video and reused
# where they fit, reads better than two dozen near-duplicates.
def _pool_target(n_lines: int) -> int:
    """Pool size scaled to the video, because a fixed pool cannot both avoid
    repetition and fill every line.

    Five images across a 12-line video at 3 images per line is 36 slots drawn
    from 5 pictures, so the same two images necessarily come back constantly --
    that was the user's complaint. Roughly half the demanded slots is enough to
    alternate without visible looping, and every image in the pool is now
    event-anchored and metadata-validated, so a bigger pool no longer means the
    old "random boats" filler.
    """
    want = max(1, n_lines) * MAX_IMAGES_PER_LINE // 2
    return max(5, min(want, int(os.getenv("ASSET_MAX_POOL_IMAGES", "16"))))


MAX_POOL_ENTITIES = int(os.getenv("ASSET_MAX_POOL_ENTITIES", "6"))
# No single query may dominate the pool. Measured: 3 of 5 images came from
# "USS Akagi Battle of Midway" and the rest were near-identical carrier
# portraits, so the matcher had three photos it could not tell apart and
# alternated the same two across the video. Diversity has to be enforced on
# the QUERY, not on the file, because the duplicates are different files.
MAX_IMAGES_PER_QUERY = int(os.getenv("ASSET_MAX_IMAGES_PER_QUERY", "2"))
# How many raw terms to pull before non-visual filtering. Filtering discards
# roughly a third of them, so the raw ask has to be comfortably larger than
# the final count or the pool ends up built from one or two survivors.
ENTITY_OVERFETCH = int(os.getenv("ASSET_ENTITY_OVERFETCH", "12"))
# How many images one line may hold. Two is enough for a line that describes a
# sequence ("launch, then the strike hits") without turning every line into a
# slideshow.
MAX_IMAGES_PER_LINE = int(os.getenv("ASSET_MAX_IMAGES_PER_LINE", "3"))


def mapping_paths(mapping: dict) -> list:
    """Every image path referenced by a {line_id: [paths]} plan, in order."""
    out = []
    for deck in mapping.values():
        for p in deck or []:
            if p not in out:
                out.append(p)
    return out


def _cap_per_term(mapping: dict, images: dict, lines: list) -> dict:
    """Enforce a maximum number of images per search term across the video.

    A rendered video came out as "a photo of an island and a bunch of white
    birds": the term "Midway" means the atoll, so Commons returned seabirds
    and beaches, and all 6 of the 6 assigned slots went to that one term. The
    pool was fine (58 images, 9 real entities) -- assignment is what failed,
    because the local planner is shown the catalog as
    "image 1: Midway ... image 6: Midway" and cannot tell six island photos
    apart from six carriers.

    Term repetition is a symptom of a term that is wrong for the video, so the
    fix is structural rather than a better prompt: no term may fill more than
    one line's worth of visual time. Overflow images are dropped rather than
    reassigned, because reassigning is what produced the birds in the first
    place; the remaining terms are the ones the planner already ranked above.

    The cap is adaptive: with many terms every line still gets a fresh visual,
    but if there are fewer terms than lines (a narrow topic, or the fallback
    path where one query covers everything) the cap is relaxed to MIN_TERMS so
    that lines are never left without an image. Never ship a line with no
    visual to avoid a repetitive one.
    """
    if not mapping or not images:
        return mapping

    # Diversity is counted over the whole candidate POOL, not over the plan.
    # The pool for a real Midway run held 58 images across 9 terms, but the
    # planner assigned only "Midway" -- counting terms in the plan saw 1 and
    # relaxed the cap back to 6, which let the seabirds through and defeated
    # the entire fix. Backfill draws from the pool, so the pool is the thing
    # that decides whether a starved line has anything better to fall back on.
    pool_terms = {(images[p][1] or images[p][0] or "").strip()
                  for p in images}
    n_lines = max(1, len([ln for ln in lines]))
    cap = max(MAX_IMAGES_PER_TERM, -(-n_lines // max(1, len(pool_terms)))) \
        if pool_terms else MAX_IMAGES_PER_TERM

    # Preserve planner intent, then walk lines in order dropping any image
    # whose term has already met its cap, and never repeating a path.
    order = [ln["id"] for ln in lines]
    seen_terms: dict[str, int] = {}
    seen_paths: set = set()
    out: dict = {}
    for lid in order + [k for k in mapping if k not in order]:
        deck = [p for p in mapping.get(lid, []) if p in images and p not in seen_paths]
        kept = []
        for p in deck:
            term = (images[p][1] or images[p][0] or "").strip()
            if seen_terms.get(term, 0) >= cap:
                print(f"    [assign] dropped {os.path.basename(p)} — term "
                      f"'{term[:28]}' already has {cap}")
                continue
            seen_terms[term] = seen_terms.get(term, 0) + 1
            seen_paths.add(p)
            kept.append(p)
        if kept:
            out[lid] = kept
        elif lid in mapping:
            out[lid] = []
    return out


# ---------------------------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------------------------

def _build_query_specs(lines: list, topic: str, query_map: dict) -> list:
    """Collect one spec per unique search query, supplemented by keywords."""
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
    return _round_robin_specs(specs, MAX_QUERY_SPECS)


def _ensure_min_images(kept: dict, specs: list, assets_dir: str) -> dict:
    """Refetch in rounds until we have at least MIN_IMAGES or give up."""
    rounds = 0
    while len(kept) < MIN_IMAGES and rounds < 3:
        rounds += 1
        print(f"  [asset] Only {len(kept)} images — refetch round {rounds}...")
        extra = _download_candidates(specs, assets_dir)
        for p, v in _describe_and_filter_candidates(extra).items():
            if p not in kept:
                kept[p] = v
        print(f"  [asset] Kept now: {len(kept)}")
    return kept


def _attach_assets_to_lines(lines: list, kept: dict, mapping: dict) -> None:
    """Write each line's assigned deck + license onto the line dicts."""
    used = set()
    for line in lines:
        deck = [p for p in mapping.get(line["id"], []) if p in kept and os.path.exists(p)]
        line["asset_paths"] = deck
        line["asset_path"] = deck[0] if deck else None
        used.update(deck)
        lic = _source_meta.get(deck[0], {}).get("license", "") if deck else ""
        if lic:
            line["image_license"] = lic
        print(f"  [asset] Line {line['id']}: {len(deck)} image(s) [{lic}]")


def _backfill_gaps(lines: list, kept: dict) -> None:
    """Opt-in only: force a visual onto every line by reusing kept images.

    Disabled by default. Forcing a picture onto a sentence that has no
    confident match is what produced the "random boats": the line is not asking
    for a carrier, so it gets a carrier anyway, and the video ends up showing
    six unrelated warships for one battle. Set ASSET_BACKFILL=1 to restore the
    old behaviour.
    """
    if os.getenv("ASSET_BACKFILL", "0") not in ("1", "true", "yes"):
        return
    pool = list(kept.keys())
    if not pool:
        return
    cursor = 0
    for i, line in enumerate(lines):
        if line.get("asset_paths"):
            continue
        prev = lines[i - 1].get("asset_paths") if i > 0 else None
        pick = prev[0] if prev else pool[cursor % len(pool)]
        cursor += 1
        line["asset_paths"] = [pick]
        line["asset_path"] = pick
        lic = _source_meta.get(pick, {}).get("license", "")
        if lic:
            line["image_license"] = lic
        print(f"  [asset] Line {line['id']}: backfilled 1 image [{lic}]")


def _topic_body(topic: str) -> str:
    """Best-effort article text for the topic, used only to widen the entity
    pool. The proper nouns usually live in the body, not the headline.

    Delegates to file_helpers.topic_body_text, which reads the research
    artifact where dump_artifact() actually writes it. Never raises."""
    try:
        return topic_body_text(topic)
    except Exception:
        return ""


def _ensure_min_pool(kept: dict, entities: list, assets_dir: str,
                     n_lines: int = 10) -> dict:
    """Refetch from the entity pool in rounds until we clear MIN_IMAGES.

    The pool is topic-level, so a round re-asks Commons for the SAME entities
    with a wider per-entity budget rather than inventing new prose queries.
    """
    rounds = 0
    while len(kept) < MIN_IMAGES and rounds < 2:
        rounds += 1
        print(f"  [asset] Only {len(kept)} images - pool refetch round {rounds}...")
        extra = _download_entities(entities, assets_dir,
                                   _pool_target(n_lines))
        for p, v in _describe_and_filter_candidates(extra).items():
            if p not in kept:
                kept[p] = v
        print(f"  [asset] Kept now: {len(kept)}")
    return kept


def fetch_assets(lines: list, topic: str = "") -> list:
    """Per-line query, OCR-clean, license-safe asset fetch (no vision)."""
    assets_dir = os.path.join(workspace_path(topic), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    print(f"  [asset] Topic: {topic}")
    print("  [asset] Extracting topic entity pool (one local call)...")
    body = _topic_body(topic)
    _set_topic_era(topic, body)
    _topic_anchor[0] = topic
    if _topic_era_year:
        print(f"  [asset] Era gate: subject ~{_topic_era_year} "
              f"(rejecting files dated after {_topic_era_year + ERA_SLACK_YEARS})")
    # Over-fetch, then filter, THEN cap. Capping first was a bug: the extractor
    # returned 4 raw terms, 3 of which were non-visual ("CV-3", "Pacific
    # Theater", "World War"), so the pool was built from a single entity.
    raw_entities = extract_entities(topic, body, max_entities=ENTITY_OVERFETCH)
    # Drop terms that name a place, command or abstract group. "Pacific Theater"
    # and "USS Pacific Fleet" are perfect matches for a Midway article and
    # return a theatre-of-operations map and a command building. Filtering here
    # costs nothing; catching them after download costs a slot in the video.
    entities = [e for e in raw_entities if not is_non_visual_entity(e)]
    if len(entities) > MAX_POOL_ENTITIES:
        print(f"  [asset] Capping {len(entities)} visual entities -> "
              f"{MAX_POOL_ENTITIES}")
        entities = entities[:MAX_POOL_ENTITIES]
    dropped = [e for e in raw_entities if is_non_visual_entity(e)]
    if dropped:
        print(f"  [asset] Dropped {len(dropped)} non-visual term(s): "
              f"{', '.join(d[:24] for d in dropped[:6])}")
    print(f"  [asset] Entities: {len(entities)}")
    for e in entities[:12]:
        print(f"       - {e}")

    if not entities:
        print("  [asset] Entity pool empty - falling back to per-line queries")
        query_map = build_search_queries(lines, topic)
        specs = _build_query_specs(lines, topic, query_map)
        print(f"  [asset] {len(specs)} queries across {len(query_map)} line(s)")
        print("  [asset] Downloading candidates (legacy per-line path)...")
        candidates = _download_candidates(specs, assets_dir)
    else:
        print(f"  [asset] Downloading pool (Commons-first, pexels={'on' if _ENABLE_PEXELS else 'off'})...")
        candidates = _download_entities(entities, assets_dir,
                                       _pool_target(len(lines)))
    print(f"  [asset] Downloaded {len(candidates)} candidate(s)")

    kept = _describe_and_filter_candidates(candidates)
    print(f"  [asset] OCR-clean, kept: {len(kept)}")

    if entities:
        kept = _ensure_min_pool(kept, entities, assets_dir, len(lines))
    else:
        kept = _ensure_min_images(kept, specs, assets_dir)

    if len(kept) < MIN_IMAGES:
        print(f"  [asset] ⚠️  Only {len(kept)} images (wanted >= {MIN_IMAGES})")

    print(f"  [asset] Assigning {len(kept)} image(s) to {len(lines)} line(s)...")
    mapping = _assign_images_to_lines(kept, lines, topic)

    _attach_assets_to_lines(lines, kept, mapping)
    _backfill_gaps(lines, kept)

    used = set()
    for line in lines:
        used.update(line.get("asset_paths", []))
    print(f"  [asset] Done — {len(used)}/{len(kept)} images used across lines")
    return lines
