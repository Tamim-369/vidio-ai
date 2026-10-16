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
import collections
import contextlib
import os
import re
import tempfile
import time
import threading
from statistics import median
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pytesseract
import numpy as np
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
# A page/collage of text in ANY language. English word-confidence alone misses
# non-Latin scripts (a Cyrillic classifieds collage scored 2.9% and sailed
# through), so we ALSO count the text boxes tesseract finds at a loose
# confidence floor: sparse-text layout analysis locates lines even when it
# cannot read them. Many boxes => a document/collage, not a photo.
ASSET_TEXT_LOOSE_CONF = int(os.getenv("ASSET_TEXT_LOOSE_CONF", "30"))
ASSET_TEXT_PAGE_BOXES = int(os.getenv("ASSET_TEXT_PAGE_BOXES", "25"))
# The coverage rule needs SEVERAL confident words, not one. A lone big box is
# usually a grid/panel/edge the OCR mis-reads as a single "word": a good
# "phone on a grid notebook" photo measured 29.6% coverage from exactly one such
# box and was being thrown away.
ASSET_TEXT_MIN_BOXES = int(os.getenv("ASSET_TEXT_MIN_BOXES", "4"))
# Perceptual (average-hash) near-duplicate guard: drops the same shot re-encoded
# or two near-identical frames from one stock shoot (same face twice in a video).
ASSET_DEDUP_HAMMING = int(os.getenv("ASSET_DEDUP_HAMMING", "5"))
# Ollama runs the VLM with `-np 1` (ONE slot). Firing N calls "concurrently" at a
# single slot does not overlap anything: it only queues them, so the last call's
# wall clock grows to N x the per-image cost (~34s here) and trips the timeout --
# and an abandoned timed-out image request WEDGES the runner for every later call
# (observed: a batch of 8 left gemma3:4b unable to answer a single image in 200s
# until the runner was restarted). So the default is strictly sequential.
ASSET_VISION_WORKERS = int(os.getenv("ASSET_VISION_WORKERS", "1"))
# Per-call wall-clock cap. Each serialized call needs ~34s; 120s leaves generous
# slack while still bounding a wedged runner to a sane retry, not a 300s stall.
ASSET_VISION_TIMEOUT = int(os.getenv("ASSET_VISION_TIMEOUT", "120"))

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
from src.agents.asset.entities import clean_entities, extract_entities
from src.agents.asset.plan import plan_shots
from src.agents.asset.validate import (
    validate_candidate, is_non_visual_entity, resolve_hull, hull_designator,
    is_portrait_image,
)
from src.agents.common.llm import VISION_MODEL, vision
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
TEXT_LOOSE_CONF = ASSET_TEXT_LOOSE_CONF          # language-agnostic box floor
TEXT_PAGE_BOXES = ASSET_TEXT_PAGE_BOXES          # box count => page/collage
TEXT_MIN_BOXES = ASSET_TEXT_MIN_BOXES            # coverage rule needs >= this
DEDUP_HAMMING = ASSET_DEDUP_HAMMING              # aHash near-dup distance
VISION_WORKERS = ASSET_VISION_WORKERS            # sequential on a -np 1 server
VISION_TIMEOUT = ASSET_VISION_TIMEOUT            # per-image wall-clock cap

_file_hashes = {}       # (workspace, hash) -> path   - dedup scoped per video
_file_hash_lock = threading.Lock()
_source_meta = {}       # path -> {"license": str, "source": str}

# --- Connection reuse ------------------------------------------------------
# Every HTTP call here used to be a bare requests.get(): a fresh TCP+TLS
# handshake each time (~1s cold vs ~0.07s warm on this box). A run issues
# dozens of Commons API calls plus dozens of image fetches, so the handshakes
# alone cost minutes. One keep-alive Session per thread keeps a warm, pooled
# connection without sharing mutable state across threads.
def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(
        total=3, connect=3, read=2, backoff_factor=0.4,
        status_forcelist=(),            # 429/5xx are handled by callers
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16,
                          max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = _new_session()
        _tls.session = s
    return s


# --- Adaptive Commons rate limiting ---------------------------------------
# Wikimedia throttles bursts of API calls with empty/HTML bodies. The old guard
# was a single global lock plus a fixed 0.35s gap: every search ran strictly
# one-at-a-time, so a run's search time was the SUM of every call. This allows
# a few searches in flight together (bounded concurrency) with a small minimum
# spacing -- and it adapts: a throttle widens the gap, a success narrows it
# back, so we stay just under the rate that trips the throttle instead of
# either crawling or getting banned.
COMMONS_MAX_INFLIGHT = int(os.getenv("ASSET_COMMONS_MAX_INFLIGHT", "4"))
COMMONS_MIN_GAP = float(os.getenv("ASSET_COMMONS_MIN_GAP", "0.08"))
COMMONS_MAX_GAP = float(os.getenv("ASSET_COMMONS_MAX_GAP", "1.5"))


class _AdaptiveLimiter:
    """Bounded-concurrency gate with an adaptive inter-request gap."""

    def __init__(self, inflight: int, min_gap: float, max_gap: float):
        self._sem = threading.BoundedSemaphore(max(1, inflight))
        self._lock = threading.Lock()
        self._min_gap = max(0.0, min_gap)
        self._max_gap = max(self._min_gap, max_gap)
        self._gap = self._min_gap
        self._next_at = 0.0

    def acquire(self):
        """Take a slot and honour the adaptive inter-request gap."""
        self._sem.acquire()
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self._gap
        if wait > 0:
            time.sleep(wait)

    def release(self):
        self._sem.release()

    @contextlib.contextmanager
    def slot(self):
        self.acquire()
        try:
            yield
        finally:
            self.release()

    def throttle(self):
        """A throttled/failed call: double the gap (capped) and pause briefly."""
        with self._lock:
            self._gap = min(self._max_gap, max(self._min_gap, self._gap) * 2.0)
            self._next_at = max(self._next_at, time.monotonic() + self._gap)

    def ok(self):
        """A successful call: ease the gap back toward the minimum."""
        with self._lock:
            self._gap = max(self._min_gap, self._gap * 0.9)


class _Cooldown:
    """A shared 'do not call until T' gate.

    When Wikimedia returns 429 every thread that retries independently sleeps
    and re-hammers, which keeps the ban alive (measured: two storming runs spent
    minutes in backoff). A single shared deadline makes all threads pause
    together, so the moment the penalty lifts we resume at full speed instead of
    staggering out of a pile-up.
    """

    def __init__(self):
        self._until = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                delay = self._until - time.monotonic()
            if delay <= 0:
                return
            time.sleep(min(delay, 1.0))

    def trip(self, seconds: float) -> None:
        with self._lock:
            self._until = max(self._until, time.monotonic() + seconds)


_COMMONS_LIMITER = _AdaptiveLimiter(COMMONS_MAX_INFLIGHT, COMMONS_MIN_GAP,
                                    COMMONS_MAX_GAP)
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# upload.wikimedia.org is a different host from the API and tolerates more
# parallelism, but not unbounded: the pool and per-line passes now run together,
# so raw thread counts would fire ~16 downloads at once and draw 429s. This caps
# in-flight downloads and widens the gap when the CDN pushes back.
DOWNLOAD_MAX_INFLIGHT = int(os.getenv("ASSET_DOWNLOAD_MAX_INFLIGHT", "8"))
_CDN_LIMITER = _AdaptiveLimiter(DOWNLOAD_MAX_INFLIGHT, 0.0, 0.5)

# requests' `timeout` is a PER-READ timeout: a server that accepts the
# connection and trickles a byte every few seconds never trips it. That is how a
# run hung for minutes at 0% CPU with 21 sockets open (measured). Each download
# therefore gets a hard wall-clock deadline, and a byte cap so an oversized
# stock original cannot dominate the pass.
DOWNLOAD_TOTAL_TIMEOUT = float(os.getenv("ASSET_DOWNLOAD_TIMEOUT", "12"))
MAX_IMAGE_BYTES = int(os.getenv("ASSET_MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))

# Shared 429 gates. Both the API and the CDN emit 429 (or plain-text bodies) on
# sustained bursts; see _Cooldown.
_COMMONS_COOLDOWN = _Cooldown()
_CDN_COOLDOWN = _Cooldown()

# Requested thumbnail width. Wikimedia renders any width that is not already
# cached ON DEMAND, which measured 0.6s / 1.4s / 5.9s for three images at
# iiurlwidth=1400 -- the single slowest part of a run. 1280 is a standard
# MediaWiki bucket (often already generated) and still exceeds the 1080px video
# width, so quality is unaffected.
COMMONS_THUMB_WIDTH = int(os.getenv("ASSET_COMMONS_THUMB_WIDTH", "1280"))

# Openverse (CC/PD index across Flickr, maxpixel, ...) is OFF by default. Its
# corpus is unvetted uploads whose titles loosely match a query, which is how a
# serious news story ended up illustrated with unrelated snapshots. Pexels is
# the stock provider now. Set ASSET_ENABLE_OPENVERSE=1 to bring it back.
_ENABLE_OPENVERSE = os.getenv("ASSET_ENABLE_OPENVERSE", "0") == "1"

# --- Run-scoped search memoization ----------------------------------------
# Several code paths ask Commons the same query more than once: the hull audit
# then the main pass, and every _ensure_min_pool refetch round re-asks the same
# entities. A short-lived cache of successful responses makes those repeats
# free. Only non-empty JSON is cached, so a throttled empty body still retries
# on the network as before. Cleared once per video at the top of fetch_assets.
_commons_cache: dict = {}
_commons_cache_lock = threading.Lock()
COMMONS_CACHE_TTL = float(os.getenv("ASSET_COMMONS_CACHE_TTL", "900"))


def _commons_cache_key(params: dict) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in (params or {}).items()))


def _commons_cache_get(key: tuple):
    with _commons_cache_lock:
        ent = _commons_cache.get(key)
    if not ent:
        return None
    ts, data = ent
    if time.time() - ts > COMMONS_CACHE_TTL:
        with _commons_cache_lock:
            _commons_cache.pop(key, None)
        return None
    return data


def _commons_cache_put(key: tuple, data: dict) -> None:
    with _commons_cache_lock:
        _commons_cache[key] = (time.time(), data)


def _commons_cache_clear() -> None:
    with _commons_cache_lock:
        _commons_cache.clear()

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


# A file whose subject is itself ancient is never out of period, however
# recently the shutter clicked: "Tiurinlinna (Räisälä) hoard objects" is a
# 2012 museum photograph of a 10th-century silver cache and it was rejected by
# the date gate that exists to keep modern scenes out of an old story.
_PERIOD_SUBJECT = re.compile(
    r"archaeolog|excavat|\bhoards?\b|\bmedieval\b|\bviking|ancient|\brelics?\b|"
    r"artifacts|artefacts|\btreasures?\b|numismat|\bcoins?\b|burials?|antiquit|"
    r"bronze age|iron age|\bruins?\b|\bshipwrecks?\b|\bshipwreck\b|"
    r"\bswords?\b|\bweapons?\b|\barmor\b|\barmour\b|\btombs?\b",
    re.IGNORECASE)


def _depicts_period_subject(ii: dict, title: str) -> bool:
    text = f"{_meta_field(ii, 'Categories')} {_meta_field(ii, 'ImageDescription')} {title}"
    return bool(_PERIOD_SUBJECT.search(text))


def _modern_scene(ii: dict, title: str) -> bool:
    """True only for machinery/scenery that cannot exist in an old story."""
    if _topic_era_year is None or _topic_era_year >= 1950:
        return False
    low = f"{_meta_field(ii, 'Categories')} {_meta_field(ii, 'ImageDescription')} {title}".lower()
    return any(m in low for m in _MODERN_MARKERS)


def _anachronistic(ii: dict, title: str) -> bool:
    """True when the file's own metadata places it far outside the topic era.

    Only fires when we know the topic's year; an unknown era passes everything
    through rather than silently emptying the pool.
    """
    if _topic_era_year is None:
        return False
    if _modern_scene(ii, title):
        return True
    if _depicts_period_subject(ii, title):
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
    # Commons CDN resets connections under parallel load, and a single dropped
    # fetch used to shrink the pool. The same host rate-limits bursts with
    # 429, so downloads go through the shared CDN limiter (bounded concurrency,
    # adaptive gap) and a 429 widens that gap instead of only backing off this
    # one call. A 429 still gets its own real backoff, not the 0.6s that suits
    # a connection reset.
    attempt = 0
    delay = 0.6
    while attempt < 4:
        _CDN_COOLDOWN.wait()
        box: dict = {}

        def _work():
            try:
                r = _session().get(url, timeout=(6, 8), headers=headers,
                                   stream=True)
                box["status"] = r.status_code
                box["ct"] = r.headers.get("Content-Type", "")
                box["ra"] = r.headers.get("Retry-After", "")
                if r.status_code == 200 and "image" in box["ct"]:
                    buf = bytearray()
                    for chunk in r.iter_content(65536):
                        buf.extend(chunk)
                        if len(buf) >= MAX_IMAGE_BYTES:
                            break
                    box["content"] = bytes(buf)
                r.close()
            except Exception as e:  # noqa: BLE001
                box["err"] = e

        # Hold the slot here (not inside the worker) so a hung fetch that we walk
        # away from still releases its concurrency token.
        _CDN_LIMITER.acquire()
        try:
            th = threading.Thread(target=_work, daemon=True)
            th.start()
            th.join(DOWNLOAD_TOTAL_TIMEOUT)
            if th.is_alive():
                # Trickling/stalled host: abandon it rather than stall the run.
                print(f"      [asset] download timeout ({DOWNLOAD_TOTAL_TIMEOUT:.0f}s) "
                      f"— skip: {os.path.basename(url)[:50]}")
                return False
        finally:
            _CDN_LIMITER.release()

        if "err" in box:
            attempt += 1
            if attempt >= 4:
                return False
            time.sleep(0.6 * attempt)
            continue
        if box.get("status") == 429:
            _CDN_LIMITER.throttle()
            try:
                wait = min(30.0, max(1.5, float(box.get("ra") or "")))
            except ValueError:
                wait = min(30.0, delay)
            attempt += 1
            if attempt >= 4:
                return False
            _CDN_COOLDOWN.trip(wait)
            delay = min(30.0, delay * 2.2)
            continue
        content = box.get("content") or b""
        if box.get("status") == 200 and content and _is_usable_image(content):
            with open(path, "wb") as f:
                f.write(content)
            _CDN_LIMITER.ok()
            return True
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
        r = _session().get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": search_term, "per_page": count * 3, "orientation": "portrait"},
            timeout=12,
        )
        photos = r.json().get("photos", [])
        # Rank by how much each photo's OWN alt text overlaps the query before we
        # download, so the on-topic result is the one we keep. Pexels ranks by
        # relevance but its tail is loose -- a literal "missing man" query has
        # returned stock of unrelated text/objects. Stable: ties keep Pexels'
        # order. Never drops anything, only reorders.
        if len(photos) > 1:
            qtok = _tokens(search_term)
            if qtok:
                photos = [p for _, p in sorted(
                    enumerate(photos),
                    key=lambda t: (-len(qtok & _tokens(t[1].get("alt", "") or "")), t[0]))]
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
    """Run a Wikimedia Commons API search through the adaptive limiter.

    Returns the parsed JSON dict ({} on failure). Wikimedia rate-limits burst
    API access with empty bodies AND HTML throttle pages, so a non-JSON body is
    treated as a throttle: the limiter widens the gap and the call retries with
    backoff. Successful responses are memoized for the run so the hull audit
    and the _ensure_min_pool rounds never re-ask the same query over the wire.

    A ConnectionError (reset by peer) is retried: during a live pool run both
    Commons and Openverse started resetting connections mid-run, which silently
    cost us whole entities (Akagi and Kaga returned zero images while working
    minutes earlier).
    """
    key = _commons_cache_key(params)
    hit = _commons_cache_get(key)
    if hit is not None:
        return hit

    data: dict = {}
    for attempt in range(3):
        _COMMONS_COOLDOWN.wait()
        try:
            with _COMMONS_LIMITER.slot():
                r = _session().get(COMMONS_API, params=params, timeout=15)
        except (requests.ConnectionError, requests.Timeout):
            _COMMONS_LIMITER.throttle()
            _COMMONS_COOLDOWN.trip(0.8 * (attempt + 1))
            continue

        json_ct = "json" in r.headers.get("Content-Type", "")
        if r.status_code == 429 or not json_ct:
            # A throttle page is HTML, not JSON. Reading it as "no results" is
            # how an entire entity silently lost its category pass: the pool
            # run printed zero [category] lines while the same queries returned
            # files minutes later. Back off and retry, widening the shared gap.
            _COMMONS_LIMITER.throttle()
            backoff = (1.5, 4.0, 10.0)[attempt]
            retry_after = (r.headers.get("Retry-After") or "").strip()
            try:
                backoff = min(30.0, max(backoff, float(retry_after)))
            except ValueError:
                pass
            # Trip the shared gate so every thread pauses with us instead of
            # each sleeping its own backoff and re-hammering the penalty.
            _COMMONS_COOLDOWN.trip(backoff)
            print(f"      [commons] throttled (HTTP {r.status_code}) — "
                  f"backing off {backoff:.0f}s")
            continue

        # JSON + 2xx is a real answer. A genuinely empty result set must be
        # returned immediately -- retrying it is what the old code avoided, and
        # re-asking three times would triple the call count for a normal miss.
        try:
            parsed = r.json()
        except ValueError:
            parsed = None
        _COMMONS_LIMITER.ok()
        if parsed:
            _commons_cache_put(key, parsed)
            return parsed
        return {}

    return data or {}


# Files a category holds that are not photographs of the subject.
_CATEGORY_NON_FILE_EXT = (".svg", ".pdf", ".djvu", ".webm", ".ogv", ".ogg",
                          ".tif", ".tiff", ".xcf", ".gif")

# A video pipeline wants rasters and nothing else. The negative list above
# misses the Commons formats that a category descent walks into first — a
# search for "Finland" ended in "Category:Audio files of Finland" (.wav/.flac)
# and handed four sound files to the image downloader.
_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _is_raster(title: str) -> bool:
    return title.lower().endswith(_IMAGE_EXT)


def _category_pages(entity: str, count: int) -> dict:
    """Files from the Commons category that best matches `entity`.

    A keyword search has to guess how the file was named; a category is the
    curator's own grouping of everything about the subject, so
    "Buffalo Bill" hands back Buffalo Bill and nothing else.

    TWO calls, not three-to-eleven. The old path found the category, listed its
    members, fetched imageinfo separately, and -- when the category was an index
    node -- walked its children up to eight more times. Measured: 3-11 API calls
    per entity, which is what tripped Wikimedia's 429 limiter (a 429 costs a
    multi-second backoff, and the whole asset stage then crawled). One
    ``generator=categorymembers`` call now returns the files AND their imageinfo
    together; an index category simply yields nothing here and the keyword
    search fallback in _fetch_for_keyword covers it.
    """
    term = (entity or "").strip()
    if not term:
        return {}
    found = _commons_search({
        "action": "query", "format": "json",
        "list": "search", "srsearch": term,
        "srnamespace": 14, "srlimit": 5,
    })
    cats = [h.get("title", "")
            for h in ((found.get("query") or {}).get("search") or [])
            if h.get("title")]
    if not cats:
        return {}
    low = term.lower()
    pick = cats[0]
    for c in cats:
        name = c.split(":", 1)[-1].strip().lower()
        if name == low or name in low or low in name:
            pick = c
            break

    gen = _commons_search({
        "action": "query", "format": "json",
        "generator": "categorymembers",
        "gcmtitle": pick, "gcmtype": "file",
        "gcmlimit": min(50, count * 6),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": COMMONS_THUMB_WIDTH,
    })
    pages = ((gen.get("query") or {}).get("pages")) or {}
    out = {}
    for pid, page in sorted(pages.items(),
                            key=lambda kv: kv[1].get("title", "")):
        if not page.get("imageinfo") or not _is_raster(page.get("title", "")):
            continue
        page = dict(page)
        page["index"] = len(out)
        out[pid] = page
    if out:
        print(f"      [category] {pick}: {len(out)} file(s)")
    else:
        print(f"      [category] {pick} has no usable files")
    return out


def _category_members(title: str, ns: int, kind: str, limit: int) -> list:
    data = _commons_search({
        "action": "query", "format": "json",
        "list": "categorymembers", "cmtitle": title,
        "cmnamespace": ns, "cmtype": kind, "cmlimit": limit,
    })
    return [m.get("title", "")
            for m in ((data.get("query") or {}).get("categorymembers") or [])
            if m.get("title")]


_MEDIA_RE = re.compile(r"\b(images?|photos?|pictures?|photographs?|"
                       r"sculptures?|objects?)\b", re.IGNORECASE)


def _child_rank(node: str, term: str) -> tuple:
    """Which branch of an index category to try first.

    The first child of Category:Finland is "by media type" and its first child
    is audio files; "photographs" sits further down. Rank media-looking names
    and names that share words with the entity above the alphabetically
    early index nodes so the walk spends its lookups on likely branches.
    """
    name = node.split(":", 1)[-1].lower()
    media = 1 if _MEDIA_RE.search(name) else 0
    term_words = {w.lower() for w in term.split() if len(w) > 3}
    overlap = len(term_words & set(name.replace(":", " ").split()))
    return (-media, -overlap, name)


def _category_files_by_descending(root: str, count: int,
                                  term: str = "") -> tuple[list, str]:
    """Breadth-first walk of a category's children for actual files.

    Two levels wide, eight lookups maximum: every call is throttled, and an
    index category is only ever a node or two above the files themselves.
    The root is already known to hold none, so the walk starts at its children.
    """
    def _files(node: str) -> list:
        return [t for t in _category_members(node, 6, "file", count * 4)
                if _is_raster(t)]

    budget = 8
    kids = sorted(_category_members(root, 14, "subcat", 6)[:6],
                  key=lambda n: _child_rank(n, term or root))[:4]
    for kid in kids:
        if budget <= 0:
            break
        budget -= 1
        files = _files(kid)
        if files:
            return files, kid
        for grand in sorted(_category_members(kid, 14, "subcat", 6)[:4],
                            key=lambda n: _child_rank(n, term or root))[:2]:
            if budget <= 0:
                break
            budget -= 1
            files = _files(grand)
            if files:
                return files, grand
    return [], root


def _category_pass(entity: str, base_path: str, count: int) -> list:
    """Category-first download pass for a proper-noun entity."""
    try:
        pages = _category_pages(entity, count)
    except Exception as e:
        print(f"      [category] Error: {e}")
        return []
    if not pages:
        return []
    return _commons_download_pages(pages, base_path, count, entity)


def _fetch_wikimedia_commons(search_term: str, base_path: str, count: int = 1) -> list:
    """Wikimedia Commons images with explicit, allowed licenses (PD/CC0/CC BY/CC BY-SA).

    Runs at most twice: once for the term as written, and -- only if the term
    carries a hull designation that Commons contradicts -- once for the
    designation Commons actually agrees with.
    """
    # Correct the hull designation BEFORE downloading: the budget gets spent on
    # the first pass, so a post-download audit would be too late to matter.
    return _commons_pass(_audited_term(search_term, count), base_path, count)


def _audited_term(search_term: str, count: int = 6) -> str:
    """Return the term with a hull number Commons agrees with, else unchanged.

    The audit is a metadata-only search (no image bytes) and is skipped for
    terms with no hull designation, so the common path costs nothing extra.
    It searches with the same `count` the main pass will use, so when a term has
    no boilerplate to strip the main pass reads the audit's response straight
    from the run cache instead of re-asking Commons.
    """
    if not hull_designator(search_term):
        return search_term
    try:
        data = _commons_search(_commons_params(search_term, count))
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


# Words the query agent adds for a human reader that Commons must not see.
# "Japanese carrier Akagi burning aerial view 1942 archive photo" is 9 ANDed
# terms and matches almost nothing; the same query without "archive photo" and
# "historical photograph" returns the photographs. Measured: 18 per-line
# queries produced 5 candidates before this, and the boilerplate was in 17 of 18.
_QUERY_BOILERPLATE = {
    "archive", "archived", "photo", "photos", "photograph", "photographs",
    "historical", "history", "period", "close-up", "closeup", "close",
    "image", "images", "picture", "pictures", "colorized", "colourised",
    "colorised", "black", "white", "file", "files", "public", "domain",
    "high", "resolution", "high-resolution", "original", "vintage",
}


def _commons_query(term: str) -> str:
    """Shorten a written query into something Commons will actually match."""
    words = (term or "").split()
    kept = [w for w in words
            if w.strip(".,;:()\"'").lower() not in _QUERY_BOILERPLATE]
    if not kept:
        kept = words
    # Commons ANDs every term, so a long query collapses to zero hits.
    trimmed = kept[:6]
    # Truncation can leave a dangling function word ("...launching from"),
    # which Commons then requires literally.
    while trimmed and trimmed[-1].strip(".,;:()\"'").lower() in {
            "from", "to", "on", "in", "at", "of", "with", "and", "or", "the",
            "a", "an", "for", "by", "as", "into", "over", "near"}:
        trimmed.pop()
    return " ".join(trimmed)


def _commons_params(search_term: str, count: int) -> dict:
    return {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {_commons_query(search_term)}",
            "gsrnamespace": 6,
            "gsrlimit": count * 4,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size",
            "iiurlwidth": COMMONS_THUMB_WIDTH,
    }


def _commons_pages_with_backoff(search_term: str, count: int) -> dict:
    """Search, and on a miss retry with progressively fewer words.

    Commons ANDs every term, so a descriptive query like "Japanese carrier
    deck crew rearming aircraft" matches nothing at all -- no file uses all
    seven words. "Japanese carrier deck crew" matches some. Short concrete
    queries ("SBD Dauntless launching") are unaffected and cost one search;
    the extra searches are only paid on a miss.
    """
    seen_variants: set = set()
    terms = _commons_query(search_term).split()
    # Never back off below three words. "US Navy carrier Enterprise 1942"
    # shortening to "US Navy" returns generic fleet photography -- worse than
    # no image at all, because it looks plausible and matches nothing in the
    # narration. A three-word AND is specific enough to stay on topic.
    floor = min(3, len(terms))
    variants = [" ".join(terms[:n]) for n in
                range(len(terms), floor - 1, -1)]
    pages: dict = {}
    for variant in variants:
        if not variant:
            break
        if variant in seen_variants:
            continue
        seen_variants.add(variant)
        params = _commons_params(variant, count)
        data = _commons_search(params)
        pages = (data.get("query", {}) or {}).get("pages", {}) or {}
        if not pages:
            # A throttled burst returns an empty body; one quieter retry.
            data = _commons_search(params)
            pages = (data.get("query", {}) or {}).get("pages", {}) or {}
        if pages:
            if variant != variants[0]:
                print(f"      [commons] 0 hits for '{variants[0]}' "
                      f"- retried shorter as '{variant}'")
            return pages
    return pages


def _commons_pass(search_term: str, base_path: str, count: int) -> list:
    """One Commons search+download pass."""
    try:
        pages = _commons_pages_with_backoff(search_term, count)
    except Exception as e:
        print(f"      [commons] Error: {e}")
        return []
    return _commons_download_pages(pages, base_path, count, search_term)


def _commons_download_pages(pages: dict, base_path: str, count: int,
                            search_term: str) -> list:
    """Validate + download from a ready-made Commons `pages` payload."""
    downloaded = []
    try:
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
                # Recorded here because this is the only point where the file
                # title and Commons categories are both available; assignment
                # needs it later to cap how often a face may reappear.
                _IS_PORTRAIT[path] = is_portrait_image(
                    page.get("title", ""), _meta_field(ii, "Categories"))
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
    """Openverse: CC/PD images from many providers with real license per result.

    A supplementary source: it lives on its own host/service, so it keeps
    producing candidates even while Wikimedia is throttling, and it is cheap to
    run in parallel with the Commons pass. It answers like a generic image
    search though, so the same guards that keep DDG honest apply — a
    relevance check on title/tags, the news-wire provider blocklist, and the
    downstream OCR + vision subject gates.
    """
    downloaded = []
    results = []
    for attempt in range(2):
        try:
            r = _session().get(
                "https://api.openverse.org/v1/images/",
                params={
                    "q": search_term,
                    "license": "by,by-sa,cc0,pdm",
                    "license_type": "commercial",
                    "page_size": count * 4,
                },
                timeout=15,
            )
            if r.status_code == 429:
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                return downloaded
            results = (r.json() or {}).get("results", [])
            break
        except Exception as e:
            print(f"      [openverse] Error: {e}")
            return downloaded

    for it in results:
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
        tags = " ".join(t.get("name", "") for t in (it.get("tags") or []))
        if not _relevance_ok(f"{it.get('title', '')} {tags}", search_term):
            continue
        lic = it.get("license") or "cc0"
        ver = it.get("license_version") or ""
        label = f"{lic} {ver}".strip()
        path = f"{base_path}_o{len(downloaded)}.jpg"
        if _download_image(url, path) and _remember(path, label, it.get("provider", "Openverse")):
            downloaded.append(path)
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

# path -> is a portrait of a person. Populated by _commons_pass.
_IS_PORTRAIT: dict = {}


def _fetch_for_keyword(keyword: str, kw_idx: int, assets_dir: str, per_kw: int,
                       prefer_category: bool = False, kind: str | None = None) -> list:
    """Fetch images for one term across providers, routed by shot kind.

    "entity"  -> a real, archived subject: Wikimedia Commons leads (its category
                 first when the term is a proper noun), Pexels only as a top-up.
    "generic" / "symbolic" -> a representative/emotional beat with no archive
                 photo: Pexels (curated, royalty-free, tagged) leads; Commons is
                 only a last resort.

    Openverse/Flickr and DDG stay OFF by default -- their loosely-tagged uploads
    are what filled a serious story with unrelated snapshots. ASSET_ENABLE_OPENVERSE=1
    or ASSET_ENABLE_GENERIC=1 restores them for A/B comparison.
    """
    base = os.path.join(assets_dir, f"kw{kw_idx}")
    paths = []

    stock_first = kind in ("generic", "symbolic")
    if stock_first:
        # No archive photo exists for this beat; a curated stock library is the
        # right source. Commons is the fallback, not the lead.
        if _ENABLE_PEXELS:
            paths = _fetch_pexels(keyword, f"{base}p", per_kw)
        if not paths:
            paths.extend(_fetch_wikimedia_commons(keyword, f"{base}c", per_kw))
        if _ENABLE_OPENVERSE and not paths:
            paths.extend(_fetch_openverse(keyword, f"{base}o", max(1, min(2, per_kw))))
    else:
        if prefer_category:
            # A proper noun has its own category: pull that FIRST, then only use
            # the keyword search to top up what the category could not supply.
            paths = _category_pass(keyword, f"{base}g", per_kw)
            if len(paths) < per_kw:
                paths.extend(_fetch_wikimedia_commons(
                    keyword, f"{base}c", per_kw - len(paths)))
        else:
            paths.extend(_fetch_wikimedia_commons(keyword, f"{base}c", per_kw))
        if _ENABLE_PEXELS and len(paths) < per_kw:
            paths.extend(_fetch_pexels(keyword, f"{base}p", per_kw - len(paths)))
        if _ENABLE_OPENVERSE:
            paths.extend(_fetch_openverse(keyword, f"{base}o",
                                          max(1, min(2, per_kw // 2))))

    if _ENABLE_GENERIC:
        paths.extend(_fetch_ddg(keyword, f"{base}d", max(1, per_kw // 2)))
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


_PLACE_HINT = re.compile(
    r"\bHarbour\b|\bHarbor\b|\bBay\b|\bGulf\b|\bAtoll\b|\bIsland\b|"
    r"\bIslands\b|\bReef\b|\bStrait\s+of\b|\bSea\b|\bOcean\b|"
    r"\bPeninsula\b|\bCoast\b|\bValley\b|\bMountain\b|\bRiver\b|"
    r"\bTheater\b|\bTheatre\b",
    re.I,
)


def _looks_like_place(entity: str) -> bool:
    """True for place-like terms. Cheap suffix/keyword test.

    Not exhaustive, and deliberately conservative: a false positive costs one
    image from a background term, whereas a false negative costs half the pool
    budget on scenery.
    """
    e = entity.strip()
    return bool(_PLACE_HINT.search(e))


def _entity_budget(entity: str) -> int:
    """How many images a term may contribute.

    A rare, multi-word, or ship-named term earns the full budget. A single
    common noun earns 1, because the audit showed a generic term spends the
    entire image allowance on unusable results and starves the good terms.
    """
    # A PLACE is a container, not one of the things that made the event up, so
    # it can only ever yield scenery -- and for a war topic that scenery is
    # beaches, atolls and memorials. Three of seven Midway entities were places
    # (Pearl Harbor, Coral Sea, Midway Atoll), so half the pool budget went to
    # background. Places get a single image and no event anchor.
    if _looks_like_place(entity):
        return 1
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
    # Only a NAMED subject earns the event anchor. Gluing the headline onto a
    # generic term produced searches nobody could match ("Ireland Resolved.
    # Patrick O'Sullivan disappeared"): a plain common noun has no event focus
    # to add, and the anchor text only pushes it further from any file title.
    if not any(w[:1].isupper() or any(ch.isdigit() for ch in w)
               for w in entity.split()):
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
        futs = {ex.submit(_fetch_for_keyword, e, i, assets_dir, budget,
                          prefer_category=True): e
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
    """Take at most max_specs specs while keeping every line's queries
    interleaved, so no line starves on download budget.

    Specs are (query, line_id[, kind]) tuples; the kind, when present, is
    carried through untouched so the fetcher can route by source."""
    if len(specs) <= max_specs:
        return specs
    queues = {}
    for spec in specs:
        queues.setdefault(spec[1], []).append(spec)
    out, idx = [], 0
    while len(out) < max_specs:
        progressed = False
        for lid in queues:
            if idx < len(queues[lid]):
                out.append(queues[lid][idx])
                progressed = True
                if len(out) >= max_specs:
                    return out
        if not progressed:
            break
        idx += 1
    return out

def _download_candidates(query_specs: list, assets_dir: str,
                         per_q: int = 0, index_offset: int = 0) -> dict:
    """Download candidates for each (query, line_id) spec in parallel.

    Returns {path: (query, line_id)} so assignment keeps the origin-line context.

    per_q overrides the shared budget, used by the per-line pass where each
    query belongs to exactly one sentence and needs a single image. `index_offset`
    shifts the file-name counter so a pass running concurrently with the entity
    pool cannot race on the same kwN path."""
    if per_q <= 0:
        per_q = max(1, int(round(TARGET_IMAGES * 1.6 / max(1, len(query_specs)))))
    candidates = {}
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(query_specs))) as ex:
        futs = {}
        for i, spec in enumerate(query_specs, start=index_offset):
            q, lid = spec[0], spec[1]
            kind = spec[2] if len(spec) > 2 else None
            futs[ex.submit(_fetch_for_keyword, q, i, assets_dir, per_q,
                           kind=kind)] = (q, lid)
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

    Two independent signals, both language-agnostic in effect:
      * coverage  — readable words (conf >= TEXT_MIN_CONF) cover more than
        MAX_TEXT_AREA of the frame: captions, memes, big watermark blocks.
      * page      — tesseract's sparse-text pass finds TEXT_PAGE_BOXES or more
        boxes at a loose confidence floor. This is what catches a document or a
        collage of text in a script the English model cannot read (a Cyrillic
        classifieds collage measured 2.9% coverage and slipped through the
        coverage rule entirely).

    Tiny credit marks stay under both. A lone large box (a grid/panel the OCR
    mis-reads as one word) stays under both as well.
    """
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
        boxes_hi = 0
        boxes_loose = 0
        for conf, txt, l, t, bw, bh in zip(
                data["conf"], data["text"], data["left"], data["top"],
                data["width"], data["height"]):
            if not (txt and txt.strip()):
                continue
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1.0
            if c >= TEXT_MIN_CONF:
                covered += float(bw) * float(bh)
                boxes_hi += 1
            if c >= TEXT_LOOSE_CONF:
                boxes_loose += 1
        if boxes_loose >= TEXT_PAGE_BOXES:
            return True
        return boxes_hi >= TEXT_MIN_BOXES and covered / area > MAX_TEXT_AREA
    except Exception:
        return False


def _perceptual_hash(path: str) -> int | None:
    """64-bit average-hash (aHash) for near-duplicate rejection.

    Exact bytes are already collapsed by `_remember`, but the same photo
    re-served at a different size (Pexels `large` vs `large2x`) or two
    near-identical frames from one stock shoot differ byte-for-byte and slip
    through. aHash collapses those: unrelated photos differ on ~half the bits,
    near-duplicates on only a few.
    """
    try:
        img = Image.open(path).convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float64)
        if arr.size == 0:
            return None
        bits = (arr >= arr.mean()).flatten()
        out = 0
        for b in bits:
            out = (out << 1) | int(b)
        return out
    except Exception:
        return None


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _drop_near_duplicates(kept: dict) -> dict:
    """Remove near-identical images, keeping the first occurrence.

    This is the guard against one woman-on-phone photo appearing on line 4 and
    line 8: those were two different files (so path/hash dedup missed them) of
    the same face. Runs before the (slow) vision pass so a duplicate never costs
    a model call.
    """
    out: dict = {}
    seen: list = []
    dropped = 0
    for p, v in kept.items():
        h = _perceptual_hash(p)
        if h is not None and any(_hamming(h, s) <= DEDUP_HAMMING for s in seen):
            dropped += 1
            print(f"      [dedup] ✗ near-duplicate — skip: {os.path.basename(p)}")
            continue
        if h is not None:
            seen.append(h)
        out[p] = v
    if dropped:
        print(f"      [dedup] removed {dropped} near-duplicate image(s)")
    return out


# Vision gate: does the picture actually show what the query named?
# Off switch: ASSET_VISION_CHECK=0 (tests and emergency renders).
ASSET_VISION_CHECK = os.getenv("ASSET_VISION_CHECK", "1").strip().lower() in (
    "1", "true", "yes", "on"
)

# A photo that shipped WITH the story (Reddit post media / article og:image).
# It is the real subject, so it skips the relevance gates and is used first.
_SOURCE_PROVIDER = "Source article"
# Stock providers whose loose tagging is what let unrelated photos through.
# For these the vision check is STRICT: an image is kept only if the model
# positively confirms it, not merely fails to deny it.
_STOCK_SOURCE_TOKENS = ("pexels", "openverse")


def _is_source_image(path: str) -> bool:
    return (_source_meta.get(path, {}).get("source") or "") == _SOURCE_PROVIDER


def _is_stock_image(path: str) -> bool:
    src = (_source_meta.get(path, {}).get("source") or "").lower()
    return any(t in src for t in _STOCK_SOURCE_TOKENS) or src.startswith("http")


# Distinct from the None "the model answered, but non-committally" sentinel: an
# infrastructure failure (timeout/transport) means we never got to judge the
# image at all, so it must NOT be treated as a strict-stock rejection.
_VISION_ERROR = object()


def _subject_of(query: str, topic: str) -> str:
    """The thing a focused query is really asking to be shown.

    ``_focused_query`` appends the story headline to a bare entity, so
    "USS Akagi Battle of Midway" is asking for the SHIP — a picture cannot
    "show" a battle, so naming the whole string would reject every correct
    frame. Only the trailing anchor words are stripped; if the query turns
    out to BE the anchor, it is kept whole.
    """
    if not query:
        return query
    anchor = _event_anchor(topic)
    if not anchor or not query.lower().endswith(anchor.lower()):
        return query
    subject = query[: len(query) - len(anchor)].strip()
    return subject or query


def _subject_matches(path: str, subject: str) -> bool | None:
    """True/False for an explicit answer.

    None means the model answered but non-committally (strict stock treats that
    as a rejection). `_VISION_ERROR` means the call itself failed (timeout or
    transport) and the image was never judged at all, so the caller keeps it —
    an infra failure must never be what empties the pool.
    """
    if not subject:
        return None
    try:
        out = vision(
            "You are a strict photo editor choosing ONE image for a factual "
            "history/mystery short video.\n"
            f"Shot to fill: {subject}\n"
            "Answer YES only if the image is a clear, literal, real photograph "
            "of exactly that shot. Answer NO if ANY of these is true:\n"
            "- it is only loosely or metaphorically related (generic filler);\n"
            "- it is a collage, a composite, or several unrelated pictures;\n"
            "- it is dominated by text/writing, a document or newspaper page, a "
            "chart, a table, or a screenshot (in any language);\n"
            "- it is a logo, an illustration, or has an obvious watermark.\n"
            "Reply with exactly one word: YES or NO.",
            path,
            num_predict=8,
            timeout=VISION_TIMEOUT,
        )
    except Exception as e:  # noqa: BLE001
        print(f"      [vision] ⚠ {os.path.basename(path)}: {e} — keeping")
        return _VISION_ERROR
    first = out.strip().lower().lstrip("*# ").split()[:1]
    if first and first[0] in ("yes", "y", "true"):
        return True
    if first and first[0] in ("no", "n", "false"):
        return False
    return None


def _vision_filter(kept: dict, topic: str) -> dict:
    """Drop images that do not show their own subject.

    Sequential by default: the Ollama VLM has a single slot (`-np 1`), so a burst
    of concurrent calls only queues them, stretches each call's wall clock to the
    whole batch, and — when one times out — leaves the runner wedged for every
    later call. One call in flight at a time is both correct and, on this box,
    faster in wall-clock terms. ASSET_VISION_WORKERS can raise it only if the
    server is reconfigured with more parallel slots.
    """
    items = list(kept.items())
    if not items:
        return kept

    def check(pair):
        p, (q, q2, lid) = pair
        if _is_source_image(p):
            # Real photo of the subject, straight from the story's source.
            return p, (q, q2, lid)
        verdict = _subject_matches(p, _subject_of(q, topic))
        if verdict is _VISION_ERROR:
            # We never got an answer (timeout/transport), so the image was not
            # judged. Keep it: a broken model must never empty the pool.
            return p, (q, q2, lid)
        # Stock is judged strictly: a loosely-tagged Pexels photo must be
        # positively confirmed, because "the model did not object" is exactly
        # how an unrelated snapshot survives. Commons keeps the permissive rule
        # (reject only an explicit NO) so a vision hiccup cannot empty a correct
        # archive pool.
        strict = _is_stock_image(p)
        if verdict is False or (strict and verdict is None):
            why = "subject mismatch" if verdict is False else "unconfirmed (strict stock)"
            print(f"      [vision] ✗ {why} — skip: {os.path.basename(p)}"
                  f" (query: {q[:44]})")
            return None
        return p, (q, q2, lid)

    workers = max(1, min(VISION_WORKERS, len(items)))
    if workers == 1:
        results = [check(pair) for pair in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(check, items))
    out = {}
    for res in results:
        if res is not None:
            out[res[0]] = res[1]
    dropped = len(items) - len(out)
    if dropped:
        print(f"      [vision] rejected {dropped}/{len(items)} image(s) as "
              f"off-subject (model {VISION_MODEL})")
    return out


def _describe_and_filter_candidates(candidates: dict, facets_tokens: set | None = None, topic: str = "") -> dict:
    """OCR text-overlay rejection, then a local VLM subject check.

    candidates: {path: (query, line_id)}. Returns {path: (query, query, line_id)}
    in insertion order; the query doubles as the image's feature text.
    The facets_tokens gate is intentionally removed: the per-line query agent
    already pins images to the topic, so pixel-level verification is gone."""
    items = list(candidates.items())

    def _filter_pair(pair):
        p, (q, lid) = pair
        # A story's own photo (e.g. a missing-person poster) legitimately carries
        # text; only stock/archive scans are judged for text overlay.
        if not _is_source_image(p) and _image_has_text_overlay(p):
            print(f"      [text] ✗ text/collage overlay — skip: {os.path.basename(p)}")
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
    # Collapse near-identical photos before the slow vision pass: the same shot
    # twice is the duplicate-face failure, and checking it costs a model call.
    kept = _drop_near_duplicates(kept)
    if kept and ASSET_VISION_CHECK:
        # Only stock is sent to the VLM. Wikimedia files already passed the
        # deterministic metadata/era/subject validation, and vision on this CPU
        # box is the single slowest step (a serialized `-np 1` queue measured up
        # to 300s per call). Stock -- loosely-tagged Pexels photos -- is exactly
        # where an unrelated image slips through, so that is what gets checked,
        # and strictly.
        reviewed = _vision_filter(
            {p: v for p, v in kept.items() if _is_stock_image(p)}, topic)
        kept = {p: v for p, v in kept.items()
                if (not _is_stock_image(p)) or p in reviewed}
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

    # How often one picture may appear in the whole video. A 6-image pool fills
    # ~24 slots in a 10-line video, so without a cap every picture is reused
    # about four times -- and a portrait's face then appears four times, which
    # is what was reported ("dude's face way more than it should be").
    # Portraits get a tighter cap because a face is the most arresting image
    # available; one appearance is enough to establish who someone is.
    uses = collections.Counter()
    limits = {p: (MAX_USES_PER_PORTRAIT if _IS_PORTRAIT.get(p)
                  else MAX_USES_PER_IMAGE) for p in images}

    def room_for(p: str) -> bool:
        return uses[p] < limits[p]

    def score_of(p: str, lid) -> int:
        feats = _tokens(images[p][0] or images[p][1] or "")
        return len(feats & want.get(lid, set()))

    # Phase 0: images fetched FOR a line win that line outright. This is what makes
    # "a line about dive bombers shows a dive bomber" true rather than hoped
    # for: token overlap between a narration line and a topic-level entity query
    # is weak by construction, so the line about aircraft scored zero and
    # shipped bare while the one line naming a carrier got everything.
    for lid in order:
        for p, meta in images.items():
            if len(out[lid]) >= MAX_IMAGES_PER_LINE or not room_for(p):
                continue
            if meta[2] != lid:              # fetched for a different line
                continue
            if score_of(p, lid) <= 0 and meta[0] is None:
                continue
            out[lid].append(p)
            uses[p] += 1
            placed.add(p)

    # Phase 1: give every image its single best-matching line, no repeats. This
    # is what stops the pool collapsing -- scoring every (image, line) pair and
    # filling to the cap sent the same best image to nine lines and left 3 of 5
    # images unused in the render.
    for score, lid, p in scores:
        if score <= 0 or p in placed or not room_for(p):
            continue
        if len(out[lid]) >= MAX_IMAGES_PER_LINE:
            continue
        out[lid].append(p)
        placed.add(p)
        uses[p] += 1

    # Phase 2: reuse ONLY to keep a line from being bare. A picture shown on a
    # second line reads to the viewer as a mistake ("the same photo twice"),
    # which is precisely what was reported, so a line that already has any
    # visual never takes a repeat. Reuse is the last resort for a line that
    # would otherwise be empty; prefer the least-used matching image so the
    # fallback spreads out instead of leaning on one picture.
    for score, lid, p in sorted(scores, key=lambda t: (uses[t[2]], -t[0], t[1])):
        if out[lid] or score <= 0 or not room_for(p):
            continue                      # never repeat onto a line that has a visual
        idx = order.index(lid)
        if idx and out.get(order[idx - 1]) and out[order[idx - 1]][-1] == p:
            continue                      # same image as the previous line
        out[lid].append(p)
        uses[p] += 1

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
# How many times one picture may appear in a whole video. Without a cap a small
# pool is recycled until every image is on screen four times.
MAX_USES_PER_IMAGE = int(os.getenv("ASSET_MAX_USES_PER_IMAGE", "2"))
# A face is the most arresting image available, so it is held to one appearance.
MAX_USES_PER_PORTRAIT = int(os.getenv("ASSET_MAX_USES_PER_PORTRAIT", "1"))


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
        kind = ln.get("asset_kind")
        for q in ln.get("search_queries", []) or []:
            if q and q not in seen:
                seen.add(q)
                specs.append((q, ln["id"], kind))
    if len(specs) < 4:
        print("  [asset] Query agent thin - supplementing with topic keywords")
        for kw in _generate_topic_keywords(topic):
            if kw not in seen:
                seen.add(kw)
                specs.append((kw, None, None))
    return _round_robin_specs(specs, MAX_QUERY_SPECS)


def _ensure_min_images(kept: dict, specs: list, assets_dir: str,
                       topic: str = "") -> dict:
    """Refetch in rounds until we have at least MIN_IMAGES or give up."""
    rounds = 0
    while len(kept) < MIN_IMAGES and rounds < 3:
        rounds += 1
        print(f"  [asset] Only {len(kept)} images — refetch round {rounds}...")
        extra = _download_candidates(specs, assets_dir)
        for p, v in _describe_and_filter_candidates(extra, topic=topic).items():
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
                     n_lines: int = 10, topic: str = "") -> dict:
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
        for p, v in _describe_and_filter_candidates(extra, topic=topic).items():
            if p not in kept:
                kept[p] = v
        print(f"  [asset] Kept now: {len(kept)}")
    return kept


_vision_warmed = threading.Event()


def _warm_vision_async() -> None:
    """Load the vision model in the background while the network phase runs.

    The first vision call to an unloaded model pays the full weight load
    (measured ~40s for gemma3:4b here, then 0.4s per image). Firing a throwaway
    call at the start of fetch_assets overlaps that load with entity extraction
    and downloading, so by the time _vision_filter runs the model is resident
    and keep_alive holds it there for the rest of the batch.
    """
    if _vision_warmed.is_set() or not ASSET_VISION_CHECK:
        return
    _vision_warmed.set()

    def _run():
        with contextlib.suppress(Exception):
            probe = os.path.join(tempfile.gettempdir(), "videoai_vision_warmup.jpg")
            Image.new("RGB", (64, 64), (0, 0, 0)).save(probe)
            vision("warmup", probe, num_predict=1)

    threading.Thread(target=_run, daemon=True).start()


def _apply_plan(plan: dict, lines: list) -> None:
    """Write the planner's per-beat decisions onto the lines.

    Sets ``search_queries`` (the query the fetcher runs) and ``asset_kind``
    (the source routing hint) for every line the plan covers. Lines the planner
    skipped keep whatever the query agent / lab already gave them.
    """
    shots = plan.get("shots") or {}
    for ln in lines:
        shot = shots.get(ln.get("id"))
        if shot is None:
            shot = shots.get(str(ln.get("id")))
        if not shot:
            continue
        if shot.get("queries"):
            ln["search_queries"] = list(shot["queries"])
        ln["asset_kind"] = shot.get("kind")
        if shot.get("intent"):
            ln["image_intent"] = shot["intent"]


def _download_source_images(urls: list, assets_dir: str) -> dict:
    """Download the real photos that shipped with the story's source.

    These are the true images of the subject (Reddit post media, article
    og:image / in-body photos). They bypass the relevance + OCR gates and are
    preferred in assignment, so a private-person story shows the actual person
    instead of a stand-in. Returns {path: (query, line_id)}.
    """
    candidates: dict = {}
    urls = [u for u in (urls or []) if u]
    if not urls:
        return candidates

    def work(i: int, url: str):
        path = os.path.join(assets_dir, f"source{i}.jpg")
        try:
            if _download_image(url, path) and _remember(
                    path, "Source article image", _SOURCE_PROVIDER):
                return path
        except Exception:  # noqa: BLE001
            return None
        return None

    with ThreadPoolExecutor(max_workers=min(4, len(urls))) as ex:
        futs = [ex.submit(work, i, u) for i, u in enumerate(urls)]
        for fut in as_completed(futs):
            p = fut.result()
            if p:
                candidates[p] = (_SOURCE_PROVIDER, None)
    return candidates


def _prefer_source_images(mapping: dict, lines: list, kept: dict, topic: str) -> dict:
    """Put the story's own photos on the beats that name its subject.

    Source images are the genuine article, so they lead the deck on the lines
    that actually mention the subject (by token overlap with the topic), and any
    left over fill a line that otherwise has no image at all.
    """
    src = [p for p in kept if _is_source_image(p)]
    if not src:
        return mapping
    used = {p for deck in mapping.values() for p in deck}
    pool = [p for p in src if p not in used]
    if not pool:
        return mapping
    subj = _tokens(topic)

    def named(ln) -> int:
        txt = " ".join([
            ln.get("text") or "", ln.get("image_intent") or "",
            " ".join(ln.get("search_queries") or []),
        ]).lower()
        return len(subj & set(re.findall(r"[a-z0-9']{3,}", txt)))

    order = sorted(lines, key=lambda ln: (-named(ln), ln["id"]))
    for ln in order:                       # subject beats: real photo goes first
        if not pool:
            break
        if named(ln) > 0:
            mapping.setdefault(ln["id"], []).insert(0, pool.pop(0))
    for ln in order:                       # then cover any still-bare line
        if not pool:
            break
        if not mapping.get(ln["id"]):
            mapping[ln["id"]] = [pool.pop(0)]
    if pool:
        mapping.setdefault(lines[0]["id"], []).extend(pool)
    return mapping


def fetch_assets(lines: list, topic: str = "",
                 source_images: list | None = None) -> list:
    """Per-line query, OCR-clean, license-safe asset fetch (no vision)."""
    assets_dir = os.path.join(workspace_path(topic), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    print(f"  [asset] Topic: {topic}")
    # A fresh video starts with a clean search cache (the previous video's
    # Commons responses must not leak into this one) and the vision model
    # loading in the background.
    _commons_cache_clear()
    _warm_vision_async()
    # Plan the shots in the background. The planner reads the topic + script
    # once and decides, per beat, what to show and whether it is a real archived
    # subject (Commons) or a generic/symbolic one (stock) -- and which named
    # subjects actually exist in an archive. When it fails, we fall back to the
    # old entity extractor + query agent, so a bad plan is never worse than none.
    _qmap_holder: dict = {}

    def _build_q():
        try:
            _qmap_holder["plan"] = plan_shots(topic, lines)
        except Exception as e:  # noqa: BLE001
            _qmap_holder["plan_e"] = e
        if not _qmap_holder.get("plan"):
            try:
                _qmap_holder["q"] = build_search_queries(lines, topic)
            except Exception as e:  # noqa: BLE001
                _qmap_holder["q_e"] = e

    _qmap_thread = threading.Thread(target=_build_q, daemon=True)
    _qmap_thread.start()

    body = _topic_body(topic)
    _set_topic_era(topic, body)
    _topic_anchor[0] = topic

    _qmap_thread.join()
    plan = _qmap_holder.get("plan")
    qmap = _qmap_holder.get("q") or {}
    if _qmap_holder.get("plan_e"):
        print(f"  [asset] Planner error: {_qmap_holder['plan_e']}")
    if _qmap_holder.get("q_e"):
        print(f"  [asset] Query agent unavailable: {_qmap_holder['q_e']}")

    if plan:
        _apply_plan(plan, lines)
        if plan.get("year"):
            global _topic_era_year
            _topic_era_year = plan["year"]
        print(f"  [asset] Plan: mode={plan.get('mode')} "
              f"({len(plan.get('real_subjects') or [])} real subject(s), "
              f"{len(plan.get('shots') or {})} planned shot(s))")
        # The entity pool comes ONLY from the planner's real, archival subjects.
        # A private person or an idea has no Commons files, so it is not here --
        # those beats route to stock imagery instead.
        raw_entities = clean_entities(plan.get("real_subjects") or [], topic, body,
                                      max_entities=ENTITY_OVERFETCH)
    else:
        print("  [asset] Extracting topic entity pool (one local call)...")
        raw_entities = extract_entities(topic, body, max_entities=ENTITY_OVERFETCH)

    if _topic_era_year:
        print(f"  [asset] Era gate: subject ~{_topic_era_year} "
              f"(rejecting files dated after {_topic_era_year + ERA_SLACK_YEARS})")
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
        specs = _build_query_specs(lines, topic, qmap)
        print(f"  [asset] {len(specs)} queries across {len(qmap)} line(s)")
        print("  [asset] Downloading candidates (legacy per-line path)...")
        candidates = _download_candidates(specs, assets_dir)
    else:
        print(f"  [asset] Downloading pool + per-line queries "
              f"(Commons-first, pexels={'on' if _ENABLE_PEXELS else 'off'})...")
        # Per-line queries run ALONGSIDE the topic pool, not after it. The pool
        # is focused but topic-level: a narration line about dive bombers shares
        # no tokens with "USS Akagi", so that line scored zero and shipped bare.
        # The query agent reads the SCRIPT and asks what should be pictured on
        # each line. Running the two passes together means their searches and
        # downloads interleave through the same adaptive limiter, so the wall
        # time is the max of the two rather than the sum -- and the per-line pass
        # no longer starts only after the pool has drained the CDN's burst
        # allowance (which used to return nothing for a good query).
        line_specs = _build_query_specs(lines, topic, qmap)
        per_line: dict = {}
        with ThreadPoolExecutor(max_workers=2) as phase_ex:
            ent_fut = phase_ex.submit(_download_entities, entities, assets_dir,
                                      _pool_target(len(lines)))
            line_fut = (phase_ex.submit(_download_candidates, line_specs,
                                        assets_dir, 1, 1000)
                        if line_specs else None)
            candidates = ent_fut.result()
            if line_fut is not None:
                try:
                    per_line = line_fut.result()
                except Exception as e:
                    print(f"  [asset] Per-line query pass unavailable: {e}")
        print(f"  [asset] Per-line queries: {len(line_specs)} query/ies "
              f"-> {len(per_line)} candidate(s) matched to a line")
        for p, meta in per_line.items():
            candidates.setdefault(p, meta)
    print(f"  [asset] Downloaded {len(candidates)} candidate(s)")

    # The story's own photos ride along with the topic. They are the true images
    # of the subject, so they are added regardless of plan/kind and skip the
    # relevance gates downstream.
    src_candidates = _download_source_images(list(source_images or []), assets_dir)
    if src_candidates:
        print(f"  [asset] Source images: {len(src_candidates)} real photo(s) "
              f"from the story's own source")
        for p, meta in src_candidates.items():
            candidates.setdefault(p, meta)

    kept = _describe_and_filter_candidates(candidates, topic=topic)
    print(f"  [asset] OCR + subject check, kept: {len(kept)}")

    if entities:
        kept = _ensure_min_pool(kept, entities, assets_dir, len(lines),
                               topic=topic)
    else:
        kept = _ensure_min_images(kept, specs, assets_dir, topic=topic)

    if len(kept) < MIN_IMAGES:
        print(f"  [asset] ⚠️  Only {len(kept)} images (wanted >= {MIN_IMAGES})")

    print(f"  [asset] Assigning {len(kept)} image(s) to {len(lines)} line(s)...")
    mapping = _assign_images_to_lines(kept, lines, topic)
    mapping = _prefer_source_images(mapping, lines, kept, topic)

    _attach_assets_to_lines(lines, kept, mapping)
    _backfill_gaps(lines, kept)

    used = set()
    for line in lines:
        used.update(line.get("asset_paths", []))
    print(f"  [asset] Done — {len(used)}/{len(kept)} images used across lines")
    return lines
