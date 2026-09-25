"""Polite HTTP fetching with a small on-disk cache.

Every collector goes through `fetch()` so network behavior is uniform:
a desktop browser User-Agent, gzip, a hard timeout, one retry with backoff,
and per-URL disk caching under topics/_cache/ so daily runs do not re-download
unchanged pages. TTL is per-request (news wants hours, archives want weeks).
"""
from __future__ import annotations

import hashlib
import os
import time

import requests

CACHE_DIR = os.path.join("topics", "_cache")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "sec-ch-ua": '"Not/A)Brand";v="99", "Google Chrome";v="126", "Chromium";v="126"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
}


def _cache_path(url: str) -> str:
    h = hashlib.sha256(url.encode()).hexdigest()[:20]
    return os.path.join(CACHE_DIR, h)


def _read_cache(url: str, ttl: float) -> str | None:
    try:
        path = _cache_path(url)
        if not os.path.exists(path):
            return None
        if time.time() - os.path.getmtime(path) > ttl:
            return None
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _write_cache(url: str, text: str) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(url), "w", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass


def fetch(url: str, ttl: float = 6 * 3600, timeout: int = 20, streaming_text: bool = False) -> str:
    """Return the text body of `url`, using a cache when fresh.

    Raises requests.RequestException on persistent failure; callers catch and
    degrade gracefully (a dead source must not kill the agent).
    """
    cached = _read_cache(url, ttl)
    if cached is not None:
        return cached

    last_exc = None
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
            resp.raise_for_status()
            text = resp.text
            _write_cache(url, text)
            return text
        except requests.RequestException as e:
            last_exc = e
            time.sleep(1 + attempt)
    raise last_exc


def fetch_json(
    url: str,
    ttl: float = 6 * 3600,
    timeout: int = 30,
    params: dict | None = None,
) -> dict:
    import json
    from urllib.parse import urlencode

    target = url
    if params:
        target = url + ("&" if "?" in url else "?") + urlencode(params)
    text = fetch(target, ttl=ttl, timeout=timeout)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # The cache only stores bodies; a JSON endpoint that returned HTML
        # under an error page should not poison the cache for later runs.
        return {}