"""Blank-prompt topic generator (LLM brainstorm + Wikipedia + channel mining).

This is the original ``run_topic_generation`` pipeline: brainstorm viral seeds
with the LLM seam, mine competitor channels for proven-viral topics, backstop with a
Wikipedia crawl, and optionally pull Reddit — then filter, dedupe against
everything already made, rank, and save the batch queue. Kept alongside the
newer sourced agent (``agent.py``); both share ``helpers.py``.
"""
import json
import os
import time

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from dotenv import load_dotenv

load_dotenv()

# Reddit OAuth (optional topic source). Create a "script" app at
# https://www.reddit.com/prefs/apps and fill these in .env.
# Permanent auth: no cookie refresh needed, 100 requests/min free.
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME", "")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD", "")

from src.agents.topic.gemini import brainstorm_topics
from src.agents.topic.helpers import (
    BROWSER_UA,
    COOKIE_FILE,
    REDDIT_SUBREDDITS,
    TARGET_TOPICS,
    _is_duplicate,
    _load_used,
    _passes_filters,
    _rank_key,
    _save_used,
    _scan_made_videos,
    _to_topic,
)
from src.agents.topic.miner import fetch_channel_candidates
from src.agents.topic.wikipedia import (
    _verify_on_wikipedia,
    fetch_wikipedia_candidates,
)


# Which subs get the (optional) search pass. Empty = no search (top-posts only).
SEARCH_SUBS = []


def _session() -> requests.Session:
    """Requests session authenticated with the browser cookie file."""
    s = requests.Session()
    s.headers.update({"User-Agent": BROWSER_UA, "Accept": "application/json"})
    if os.path.exists(COOKIE_FILE):
        s.headers["Cookie"] = open(COOKIE_FILE).read().strip()
    s.verify = False
    return s


def _get_oauth_token() -> str:
    """Get a Reddit OAuth access token (script app, password grant).

    Permanent auth: no cookie refresh needed. Returns '' if credentials
    are not configured.
    """
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USERNAME and REDDIT_PASSWORD):
        return ""

    r = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
        data={
            "grant_type": "password",
            "username": REDDIT_USERNAME,
            "password": REDDIT_PASSWORD,
        },
        headers={"User-Agent": BROWSER_UA},
        timeout=20,
    )
    if r.status_code == 200:
        return r.json().get("access_token", "")
    print(f"    [reddit] OAuth token failed: HTTP {r.status_code} {r.text[:80]}")
    return ""


def _get_json(session: requests.Session, url: str) -> dict:
    for attempt in range(2):
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
            print(f"    [reddit] HTTP {r.status_code} — retrying...")
        except requests.RequestException as e:
            print(f"    [reddit] {e} — retrying...")
        time.sleep(3)
    return {}


def _parse_listing(data: dict, subreddit: str) -> list:
    """Extract candidates from a Reddit Listing JSON payload."""
    candidates = []
    for child in data.get("data", {}).get("children", []):
        d = child.get("data", {})
        selftext = (d.get("selftext") or "").strip()

        if d.get("stickied"):
            continue
        if selftext in ("[removed]", "[deleted]"):
            continue
        if d.get("removed_by_category"):
            continue

        candidates.append({
            "source": f"reddit:{subreddit}",
            "title": (d.get("title") or "").strip(),
            "content": selftext or (d.get("title") or "").strip(),
            "score": d.get("score", 0),
            "upvote_ratio": d.get("upvote_ratio", 0.0),
            "source_urls": [
                f"https://www.reddit.com{d.get('permalink', '')}"
                if d.get("permalink") else d.get("url", "")
            ],
        })
    return candidates


def fetch_reddit(session: requests.Session, subreddit: str, limit: int = 100) -> list:
    url = f"https://www.reddit.com/r/{subreddit}/top.json?t=all&limit={limit}"
    data = _get_json(session, url)
    if not data:
        print(f"    [reddit] r/{subreddit}: no data")
        return []

    candidates = _parse_listing(data, subreddit)
    print(f"    [reddit] r/{subreddit}: {len(candidates)} raw posts")
    return candidates


def fetch_reddit_oauth(token: str, subreddit: str, limit: int = 100) -> list:
    """Fetch top posts via the official OAuth API (not the blocked public endpoint)."""
    url = f"https://oauth.reddit.com/r/{subreddit}/top?t=all&limit={limit}"
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "User-Agent": BROWSER_UA},
        timeout=20,
        verify=False,
    )
    if r.status_code != 200:
        print(f"    [reddit] oauth r/{subreddit}: HTTP {r.status_code}")
        return []

    candidates = _parse_listing(r.json(), subreddit)
    print(f"    [reddit] oauth r/{subreddit}: {len(candidates)} raw posts")
    return candidates


def _prime_browser(page) -> bool:
    """Homepage visit so fetches look like a genuine logged-in session."""
    try:
        page.goto("https://www.reddit.com/", timeout=25000, wait_until="domcontentloaded")
        return True
    except Exception as e:
        print(f"    [reddit] browser prime failed ({str(e)[:50]}) — using requests instead")
        return False


def _fetch_sub(page, sub: str, limit: int, candidates: list) -> None:
    """Same-origin fetch of one subreddit's top posts, extending `candidates`."""
    try:
        data = page.evaluate(
            """async (path) => {
                const r = await fetch(path);
                if (r.status !== 200) return { status: r.status, data: null };
                return { status: r.status, data: await r.json() };
            }""",
            f"/r/{sub}/top.json?t=all&limit={limit}",
        )
        if data["status"] == 200 and data["data"]:
            parsed = _parse_listing(data["data"], sub)
            candidates.extend(parsed)
            print(f"    [reddit] r/{sub}: +{len(parsed)} raw posts")
        else:
            print(f"    [reddit] r/{sub}: HTTP {data['status']}")
    except Exception as e:
        print(f"    [reddit] r/{sub}: browser fetch failed ({str(e)[:50]})")
    time.sleep(1)


def fetch_reddit_playwright(subreddits: list, limit: int = 100) -> list:
    """Fetch all subreddits through a real Chromium instance.

    Launches a headless browser, injects the browser cookie, and performs
    same-origin fetches from inside the page. This passes Reddit's bot
    detection because it looks like a genuine logged-in browser session.

    Never raises — returns [] on any failure so callers can fall back.
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("    [reddit] playwright not installed — falling back to requests")
        return []

    cookie = None
    if os.path.exists(COOKIE_FILE):
        cookie = open(COOKIE_FILE).read().strip()

    candidates = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path="/usr/bin/chromium")
            ctx = browser.new_context(
                user_agent=BROWSER_UA,
                locale="en-US",
                extra_http_headers={"Cookie": cookie} if cookie else {},
            )
            page = ctx.new_page()

            # Prime the session on the homepage first. If this fails, bail out
            # cleanly and let the caller fall back to requests.
            if not _prime_browser(page):
                browser.close()
                return []

            for sub in subreddits:
                _fetch_sub(page, sub, limit, candidates)

            browser.close()
    except Exception as e:
        print(f"    [reddit] playwright failed ({str(e)[:60]}) — using requests instead")

    return candidates


def _collect_raw_sources(limit: int, target: int, used: list, use_browser: bool) -> list:
    """Gather raw candidates from the LLM seam, channel mining, Wikipedia, Reddit."""
    raw = []

    # 1) LLM brainstorm: fresh viral topic seeds.
    print("Brainstorming viral topics with the LLM seam...")
    gemini_seeds = brainstorm_topics(count=target, blocklist=used)
    verified = []
    for s in gemini_seeds:
        if _is_duplicate(s["title"], used):
            continue
        if not _verify_on_wikipedia(s["title"]):
            print(f"  [llm] dropped (not on Wikipedia): {s['title']}")
            continue
        verified.append(s)
    for s in verified:
        raw.append({
            "source": "llm",
            "category": s["angle"],
            "title": s["title"],
            "content": s["summary"],
            "score": 80,  # grounded proposal; below channel-mined views
            "upvote_ratio": 1.0,
            "source_urls": [],
        })
    print(f"  [llm] {len(verified)}/{len(gemini_seeds)} seeds verified")

    # 2) Competitor-channel mining: proven-viral topics ranked by views.
    #    Cached on disk + parallel LLM calls, so re-runs are fast.
    print("Mining competitor channels for viral topics...")
    try:
        raw.extend(fetch_channel_candidates(used_titles=used))
    except Exception as e:
        print(f"  [miner] failed: {e}")

    # 3) Primary backstop: Wikipedia (stable, not ISP-blocked). Runs only if
    #    LLM + miner came up short, so the slow crawl is a rarity.
    target_covered = target * 3
    if len(raw) < target_covered:
        print("Pulling niche topics from Wikipedia as backstop...")
        raw.extend(fetch_wikipedia_candidates(
            limit=limit, target=target_covered - len(raw), used_titles=used))
    else:
        print("Enough from LLM + channels — skipping Wikipedia")

    # 4) Secondary: Reddit (best-effort — unreliable on this ISP).
    #    Skipped unless REDDIT_ENABLED=1 (or --limit high forces it) to keep runs fast.
    if os.environ.get("REDDIT_ENABLED") == "1":
        reddit_raw = _fetch_reddit_all(limit, use_browser)
        if reddit_raw:
            raw.extend(reddit_raw)
        else:
            print("  [reddit] skipped — Wikipedia only")

    return raw


def _fetch_reddit_all(limit: int, use_browser: bool) -> list:
    """Best-effort Reddit pull (token OAuth, or cookie requests)."""
    token = _get_oauth_token()
    session = _session()
    reddit_raw = []
    if token:
        print("  [reddit] Using Reddit OAuth API...")
        for sub in REDDIT_SUBREDDITS:
            reddit_raw.extend(fetch_reddit_oauth(token, sub, limit))
            time.sleep(1)
    else:
        print("  [reddit] Using requests + browser cookie...")
        for sub in REDDIT_SUBREDDITS:
            reddit_raw.extend(fetch_reddit(session, sub, limit))
            time.sleep(1)

    if not reddit_raw and use_browser and PLAYWRIGHT_AVAILABLE:
        print("  [reddit] Using headless Chromium + browser cookie...")
        reddit_raw = fetch_reddit_playwright(REDDIT_SUBREDDITS, limit)
    return reddit_raw


def _save_batch(topics: list) -> str:
    batch_file = os.path.join(
        "topics", f"batch_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(batch_file, "w") as f:
        json.dump(topics, f, indent=2)
    print(f"Batch saved to {batch_file}")
    return batch_file


def run_topic_generation(limit: int = 100, target: int = TARGET_TOPICS, use_browser: bool = True) -> list:
    os.makedirs("topics", exist_ok=True)
    used = list(dict.fromkeys(_load_used() + _scan_made_videos()))
    if len(used) > len(_load_used()):
        print(f"Reconciled: {len(used)} topics tracked (incl. {len(_scan_made_videos())} from output/)")

    raw = _collect_raw_sources(limit, target, used, use_browser)
    if not raw:
        print("\n  ⚠️  No data from any source.\n")
        return []

    print(f"\nTotal raw candidates: {len(raw)}")

    passed = [c for c in raw if _passes_filters(c) and not _is_duplicate(c["title"], used)]
    print(f"After filters + dedupe: {len(passed)}")

    # Rank: proven-viral channel topics first (by views), then fresh LLM
    # proposals, then Wikipedia, then Reddit.
    passed.sort(key=_rank_key, reverse=True)
    topics = [_to_topic(c) for c in passed[:target]]
    print(f"Selected: {len(topics)} topics")

    _save_used(used + [t["title"] for t in topics])
    _save_batch(topics)
    return topics