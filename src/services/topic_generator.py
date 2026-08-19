import json
import os
import re
import time
import urllib.parse
import urllib3

import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from src.config.settings import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_PASSWORD,
    REDDIT_USERNAME,
)

# Niche-focused Reddit subs (secondary source, currently unreliable on this ISP).
# Crime (serial killers, missing persons) and paranormal subs are deliberately excluded.
# Pure photo-caption subs (tankporn, WarshipPorn) are excluded too — they produce
# "tank in a garden" titles, not stories.
REDDIT_SUBREDDITS = [
    "MilitaryHistory",     # military history
    "AskHistorians",       # deep history answers
    "TheGrittyPast",       # dark / grim history
    "WeirdWings",          # experimental aircraft (great for "weird experiments")
    "WeirdWeapons",        # experimental / strange weapons
]

# Subreddits whose content is inherently contested -> flag for manual check
CONTESTED_SUBREDDITS = set()

# Crime / serial-killer / missing-person content to exclude from the niche.
CRIME_BLOCK_KEYWORDS = [
    "serial killer", "serial killer", "murder", "killer", "rapist", "stalker",
    "cctv", "kidnap", "abduction", "missing person", "body found", "death of",
    "was killed", "school shooting", "mass shooter", "investigation", "arrested",
    "police", "criminal", "manslaughter",
]

# Meta/announcement/listicle posts that aren't video topics
NON_TOPIC_PATTERNS = [
    "going dark", "shifting to", "meta:", "[meta]", "will go private",
    "policy of zero tolerance", "list of", "am i the only", "what is the best",
    "whatifalthist", "rabbit holes", "buyer of glitter", "moving to",
    "we are looking for", "state of the sub", "update on the",
    "million subscribers", "api access", "research infrastructure",
    "free fact", "recommend what to start reading", "celebrate",
    "megathread", "concerns regarding", "closing for new posts",
    "protests against", "in protest", "api changes",
    "which one is your favorite", "anyone knows what they are", "who else",
    "what is your favorite", "rate my", "oc post", "just finished", "it me",
    "tinder", "low effort", "shitpost", "meme", "repost", "colorized by me",
    "what tank is this", "what ship is this", "what aircraft is this",
    "what is this", "id this", "can anyone", "does anyone", "is this a",
    "which one", "name this", "help me identify", "looking for", "need help",
    "officially re-opened", "can't believe no-one has posted this", "i'm going to",
    "is officially", "starting things off", "reopened", "re-opened",
]

# Subs that reliably produce story-driven material -> slight ranking boost
STORY_SUBS = {
    "MilitaryHistory", "TheGrittyPast", "WeirdWings", "WeirdWeapons",
}

CATEGORY_MAP = {
    "MilitaryHistory": "weapon",
    "WeirdWings": "weapon",
    "WeirdWeapons": "weapon",
    "AskHistorians": "experiment",
    "TheGrittyPast": "mystery",
}

# ---------------------------------------------------------------- Wikipedia source
#
# Wikipedia is the primary source. It's not ISP-blocked and its API is stable.
# Each category below maps to one of the niche categories we care about.
# The provider returns candidates in the SAME dict shape as the Reddit path,
# so filtering/ranking/output are shared.

WIKI_API = "https://en.wikipedia.org/w/api.php"

# Wikipedia category -> topic category
# Story-driven only: mysteries, injustice, secrets, scary. No aircraft/weapon
# catalogs (those produce encyclopedia entries, not viral stories).
WIKI_CATEGORIES = {
    "Category:Unexplained phenomena": "mystery",
    "Category:Unexplained disappearances": "mystery",
    "Category:Missing aircraft": "mystery",
    "Category:UFO sightings": "mystery",
    "Category:Conspiracy theories": "mystery",
    "Category:Secret military programs": "mystery",
    "Category:Military scandals": "mystery",
    "Category:War crimes": "mystery",
    "Category:Massacres": "mystery",
    "Category:Human rights abuses": "mystery",
    "Category:Biological warfare": "experiment",
    "Category:Chemical warfare": "experiment",
}

# Search-driven pass (primary). These queries surface SPECIFIC story articles
# (e.g. "Unit 731", "Battalion 316", "Project MKUltra") far better than the
# category lists. Each query is one HTTP request, so this stays light.
WIKI_SEARCH_QUERIES = [
    "secret military experiment", "military mystery", "disappeared soldiers",
    "unexplained disappearance", "military cover-up", "human experimentation",
    "mind control program", "mysterious death", "classified project",
    "lost expedition", "secret underground base", "experimental aircraft program",
    "weapon test site", "psychic research", "forgotten war", "hidden history",
]

# Generic concept/list pages that aren't specific stories.
WIKI_GENERIC_BLOCKS = {
    "conspiracy theory", "war crime", "massacre", "human rights abuse",
    "biological warfare", "chemical warfare", "unexplained phenomenon",
    "list of", "disappeared person", "state terrorism", "political repression",
    "extrajudicial killing", "genocide", "mass killing", "military scandal",
    "covert operation", "psychological warfare", "chemical weapon",
    "nuclear weapon", "battle of", "campaign of", "war in", "history of",
    "human sexual activity", "human subject research", "human evolution",
    "animal testing", "human anatomy", "human body", "human genetics",
    "mental illness", "brainwashing", "mind control", "human", "human behavior",
    "classified information", "ghost hunting", "trump", "president",
    "presidential", "election", "politician", "political party",
    "enforced disappearance", "aircraft nuclear propulsion", "nuclear propulsion",
    "aircraft carrier", "propulsion", "weapons of mass destruction",
    "weapon of mass destruction", "world war",
}

# One-word generic titles that are never specific stories.
WIKI_ONE_WORD_BLOCKS = {
    "human", "brainwashing", "war", "army", "navy", "weapon", "bomb",
    "missile", "torture", "prison", "disease", "virus", "experiment",
    "science", "history", "mystery", "conspiracy", "myth", "legend",
    "soldier", "death", "massacre", "genocide", "prisoner", "disappearance",
}

# Aircraft/weapon designations (X-62, F-16, Project X) and named wars/orgs that
# read as catalogs, not stories. The user explicitly wants NO boring spec pages.
WIKI_HARD_BLOCK_PATTERNS = [
    r"\b[XFBY]-?\d+(?:-|\b)",    # X-6, F-47, YF-23, B-52
    r"\b(M\d+|T\d+)(?:-|[A-Z])", # M4 Sherman, T34
    r"\b\d+\s?-\s?\d+\b",        # range style names
]
WIKI_WAR_BLOCKS = {
    "war", "wars", "conflict", "campaign", "operation", "battle",
    "insurgency", "genocide", "revolution", "uprising",
}
WIKI_ORG_BLOCKS = {
    "network", "corporation", "company", "inc", "organization",
    "foundation", "institute", "association", "party", "union",
    "agency", "department", "division", "committee", "group",
}

# Media / pop-culture noise — movies, TV episodes, books, games, songs, fiction.
WIKI_MEDIA_BLOCKS = {
    "film", "movie", "season", "television", "tv show", "episode", "novel",
    "book", "game", "video game", "song", "album", "musical", "comic",
    "anime", "manga", "series", "franchise", "fictional", "character",
    "soundtrack", "documentary", "short film", "play", "opera", "poem",
    "short story", "magazine", "tv series", "miniseries", "film series",
    "mcU", "star wars", "captain america", "batman", "superman", "fringe",
    "garage sale mystery", "enola holmes", "cold night", "tattoo",
    "national anthem", "album", "wrestling", "sports", "video", "song",
}

# Narrative words that mark a page as a STORY (disappearance, death, conspiracy,
# experiment) rather than an encyclopedia concept. Require >= 2 in the intro.
WIKI_STORY_SIGNALS = [
    "disappeared", "disappearance", "vanished", "died", "death", "killed",
    "murdered", "found dead", "went missing", "mystery", "mysterious",
    "secret", "classified", "experiment", "experimental", "conspiracy",
    "cover-up", "cover up", "alleged", "accused", "sentenced", "arrested",
    "investigation", "never found", "unknown", "unexplained", "haunted",
    "cursed", "tortured", "executed", "massacre", "cover-up", "abducted",
    "kidnapped", "disappear", "body was", "remains", "tomb", "ancient",
    "buried", "hidden", "silence", "denied", "controversial", "scandal",
    "weaponized", "radiation", "nuclear", "chemical", "biological",
    "program", "project", "testing", "experimented",
]

# Intros that describe the article as media ("is a 2022 American film").
WIKI_MEDIA_EXTRACT_SIGNALS = [
    "is a film", "is a movie", "is a novel", "is a book", "is a game",
    "is an album", "is a song", "is a musical", "is a comic", "is an anime",
    "is a manga", "is a series", "is a miniseries", "is a documentary",
    "is a short film", "is a play", "is a poem", "is a tv", "is a television",
    "television series", "reality television", "tv series", "tv show",
    "science fiction television", "is a 20", "is a 19",
]

WIKI_HEADERS = {
    "User-Agent": "VideoAI/0.1 (faceless video topic generator; contact: local)",
    "Accept": "application/json",
}


def _wiki_get_json(params: dict) -> dict:
    params.setdefault("format", "json")
    for attempt in range(3):
        try:
            r = requests.get(
                WIKI_API, params=params, headers=WIKI_HEADERS, timeout=20, verify=False
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 5 + attempt * 5
                print(f"    [wiki] rate-limited — backing off {wait}s...")
                time.sleep(wait)
                continue
            print(f"    [wiki] HTTP {r.status_code} — retrying...")
        except requests.RequestException as e:
            print(f"    [wiki] {e} — retrying...")
        time.sleep(2)
    return {}


def _wiki_category_members(category: str, limit: int = 40) -> list:
    """Return page titles in a Wikipedia category."""
    data = _wiki_get_json({
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmnamespace": "0",
        "cmlimit": str(limit),
        "cmtype": "page",
    })
    return [m.get("title", "") for m in data.get("query", {}).get("categorymembers", []) if m.get("title")]


def _wiki_search(query: str, limit: int = 10, offset: int = 0) -> list:
    """Search Wikipedia for niche terms -> page titles (paginated)."""
    data = _wiki_get_json({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "sroffset": str(offset),
        "srnamespace": "0",
    })
    return [h.get("title", "") for h in data.get("query", {}).get("search", []) if h.get("title")]


def _wiki_extract_many(titles: list, chunk: int = 40) -> dict:
    """Fetch intro paragraphs for many articles in few batched requests."""
    results = {}
    for i in range(0, len(titles), chunk):
        batch = titles[i:i + chunk]
        data = _wiki_get_json({
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exintro": "1",
            "redirects": "1",
            "titles": "|".join(batch),
        })
        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            ext = (page.get("extract") or "").strip()
            if ext:
                results[title] = ext
        time.sleep(0.3)
    return results


def _wiki_extract(title: str) -> str:
    """Return the intro paragraph(s) of a Wikipedia article (plain text)."""
    return _wiki_extract_many([title]).get(title, "")


def fetch_wikipedia_candidates(limit: int = 40, target: int = 0, used_titles: list = None) -> list:
    """Incrementally collect niche topic candidates from Wikipedia search.

    Pages through each search query a few pages deep (Wikipedia ranks relevant
    results first), pre-filters titles by name *before* fetching intros, and
    stops as soon as we have enough. Used titles are skipped up front so re-runs
    naturally dig deeper into each query instead of returning the same topics.
    """
    used = set(t.lower() for t in (used_titles or []))
    candidates = []
    seen = set()
    target = target or limit

    def _is_generic(title: str) -> bool:
        low = title.lower()
        if any(low.startswith(p) for p in ("list of", "template:", "category:", "outline of", "index of", "timeline of")):
            return True
        if any(b in low for b in WIKI_GENERIC_BLOCKS):
            return True
        if any(b in low for b in WIKI_MEDIA_BLOCKS):
            return True
        if len(title.split()) == 1 and low in WIKI_ONE_WORD_BLOCKS:
            return True
        if re.search(r"|".join(WIKI_HARD_BLOCK_PATTERNS), title, re.IGNORECASE):
            return True
        core = re.sub(r"\s*\([^)]*\)\s*$", "", low)  # strip " (2022–present)"
        last_word = core.rsplit(" ", 1)[-1].strip("s")
        if last_word in WIKI_WAR_BLOCKS:
            return True
        if any(b in low for b in WIKI_ORG_BLOCKS):
            return True
        return False

    def _is_media_extract(extract: str) -> bool:
        low = extract.lower()
        return any(s in low for s in WIKI_MEDIA_EXTRACT_SIGNALS)

    def _candidate(title, category, extracts):
        extract = extracts.get(title, "")
        if len(extract) < 500 or _is_generic(title):
            return None
        # Skip media pages ("is a 2022 American film...").
        if _is_media_extract(extract):
            return None
        # Must read like a story, not an encyclopedia definition.
        hits = sum(1 for s in WIKI_STORY_SIGNALS if s in extract.lower())
        if hits < 2:
            return None
        return {
            "source": "wikipedia",
            "category": category,
            "title": title,
            "content": extract,
            "score": 100,
            "upvote_ratio": 1.0,
            "source_urls": [f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"],
        }

    # Search pass: page through each query until we hit target.
    for q in WIKI_SEARCH_QUERIES:
        for page in range(3):
            if len(candidates) >= target:
                break
            time.sleep(0.8)  # stay under ~1 req/s to avoid 429 backoffs
            titles = _wiki_search(q, limit=10, offset=page * 10)
            if not titles:
                break
            fresh = []
            for t in titles:
                low = t.lower()
                if low in seen or low in used or _is_generic(t):
                    seen.add(low)
                    continue
                seen.add(low)
                fresh.append(t)
            if not fresh:
                time.sleep(0.5)
                continue
            time.sleep(0.8)
            extracts = _wiki_extract_many(fresh)
            for t in fresh:
                c = _candidate(t, "mystery", extracts)
                if c:
                    candidates.append(c)
                    if len(candidates) >= target:
                        break
            time.sleep(0.3)

    # Category pass (enrichment) only if search came up short.
    if len(candidates) < target:
        for cat, category in WIKI_CATEGORIES.items():
            if len(candidates) >= target:
                break
            titles = _wiki_category_members(cat, 40)
            print(f"    [wiki] {cat}: {len(titles)} pages")
            fresh = []
            for t in titles:
                low = t.lower()
                if low in seen or low in used:
                    seen.add(low)
                    continue
                seen.add(low)
                fresh.append((t, category))
            if not fresh:
                time.sleep(0.5)
                continue
            time.sleep(0.8)
            extracts = _wiki_extract_many([t for t, _ in fresh])
            for t, category in fresh:
                c = _candidate(t, category, extracts)
                if c:
                    candidates.append(c)
                    if len(candidates) >= target:
                        break
            time.sleep(0.3)

    print(f"    [wiki] total: {len(candidates)} candidates")
    return candidates

COOKIE_FILE = "cookie"
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

USED_TOPICS_FILE = "used_topics.json"
TOPICS_OUTPUT_DIR = "topics"
TARGET_TOPICS = 20

# Filter thresholds (from spec)
MIN_SCORE = 500
MIN_SELFTEXT = 1500
MIN_UPRATIO = 0.85

# Subs that are mostly images/links (no selftext) -> lower selftext floor + lower score floor
TITLE_BASED_SUBS = {
    "WeirdWings", "WeirdWeapons",
}

# Keywords that ALONE prove the topic is on-niche: weird/scary/unknown mystery
# stories, secret experiments, dark history. (Not plain vehicle names.)
NICHE_STRONG = [
    "mystery", "mysterious", "secret", "top secret", "classified project",
    "unknown", "unexplained",
    "vanished", "disappeared", "disappear", "lost", "abandoned", "hidden",
    "curse", "cursed", "haunted", "creepy", "strange", "weird", "bizarre",
    "unsettling", "disturbing", "terrifying", "horrifying", "horror", "dark",
    "brutal", "torture", "experiment", "experimental", "laboratory", "lab",
    "chemical", "biological", "plague", "disease", "virus", "atomic",
    "nuclear", "uranium", "radiation", "prototype", "project", "test",
    "prisoner", "prison", "burial", "buried", "skeleton", "bones", "grave",
    "exhumed", "executed", "massacre", "holocaust", "legend", "myth", "mythical",
    "forgotten", "x-plane", "stealth", "bunker", "warhead", "conspiracy",
    "ancient", "medieval", "roman", "viking", "egyptian", "ruins", "artifact",
]

# Military context alone is NOT enough — a story/experiment keyword must also match.
# This list is only used as a body-signal for text subs.
NICHE_CONTEXT = [
    "ww2", "wwii", "world war", "military", "army", "navy", "aircraft",
    "tank", "fighter", "bomber", "submarine", "battleship", "weapon", "bomb",
    "missile", "nazi", "hitler", "soldier", "war", "battle", "soviet",
    "german", "japanese", "british", "american", "korea", "vietnam",
]


def _min_selftext(c: dict) -> int:
    if c.get("source") == "wikipedia":
        return 500
    sub = c["source"].split(":", 1)[-1]
    if sub in TITLE_BASED_SUBS:
        return 40
    return MIN_SELFTEXT


def _min_score(c: dict) -> int:
    if c.get("source") == "wikipedia":
        return 0
    sub = c["source"].split(":", 1)[-1]
    if sub in TITLE_BASED_SUBS:
        return 200
    return MIN_SCORE


# ---------------------------------------------------------------- auth / fetch

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


# Search queries that surface weird/scary/unknown military stories + experiments.
# Keep small — each query is a separate HTTP request and Reddit rate-limits hard.
NICHE_SEARCH_QUERIES = [
    "secret experiment", "military mystery", "disappeared", "top secret",
    "experimental weapon", "lost submarine", "ancient curse", "unexplained",
    "secret project", "vanished", "mystery", "classified",
]

# Which subs get the (optional) search pass. Empty = no search (top-posts only).
SEARCH_SUBS = []


def search_reddit(session: requests.Session, subreddit: str, limit: int = 30) -> list:
    """Search a sub for niche terms — surfaces story posts that top-lists miss."""
    found = {}
    for q in NICHE_SEARCH_QUERIES:
        url = (
            f"https://www.reddit.com/r/{subreddit}/search.json"
            f"?q={urllib.parse.quote(q)}&restrict_sr=1&sort=top&t=year&limit={limit}"
        )
        data = _get_json(session, url)
        if not data:
            continue
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            pid = d.get("id")
            if pid and pid not in found:
                found[pid] = _parse_listing({"data": {"children": [child]}}, subreddit)[0]
        time.sleep(2)
    print(f"    [reddit] r/{subreddit}: {len(found)} posts from search")
    return list(found.values())


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
            try:
                page.goto("https://www.reddit.com/", timeout=25000, wait_until="domcontentloaded")
            except Exception as e:
                print(f"    [reddit] browser prime failed ({str(e)[:50]}) — using requests instead")
                browser.close()
                return []

            for sub in subreddits:
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

            browser.close()
    except Exception as e:
        print(f"    [reddit] playwright failed ({str(e)[:60]}) — using requests instead")

    return candidates


# ---------------------------------------------------------------- filtering

def _passes_filters(c: dict) -> bool:
    if c.get("source") != "wikipedia":
        if c["score"] <= _min_score(c):
            return False
        if c["upvote_ratio"] <= MIN_UPRATIO:
            return False
    if len(c["content"]) <= _min_selftext(c):
        return False
    low_title = c["title"].lower()
    if any(p in low_title for p in NON_TOPIC_PATTERNS):
        return False
    # Hard niche gate: must be military / experiments / dark history, not crime.
    if any(b in low_title for b in CRIME_BLOCK_KEYWORDS):
        return False
    if not _is_niche(c):
        return False
    return True


def _is_niche(c: dict) -> bool:
    """A topic is on-niche only if it carries a story/experiment signal.

    Wikipedia candidates already come from niche categories, so they only need
    a body signal. Reddit title-based (image) subs must hit a strong story
    keyword in the title; text subs need strong/context body signals.
    """
    low_title = c["title"].lower()
    low_body = c["content"].lower()

    if c.get("source") == "wikipedia":
        body_hits = sum(1 for k in NICHE_STRONG if k in low_body)
        context_hits = sum(1 for k in NICHE_CONTEXT if k in low_body)
        return body_hits >= 2 or (body_hits >= 1 and context_hits >= 2)

    sub = c["source"].split(":", 1)[-1]
    if any(k in low_title for k in NICHE_STRONG):
        return True
    if sub in TITLE_BASED_SUBS:
        return False  # title-only posts must carry a story keyword

    body_hits = sum(1 for k in NICHE_STRONG if k in low_body)
    if body_hits >= 2:
        return True
    context_hits = sum(1 for k in NICHE_CONTEXT if k in low_body)
    return body_hits >= 1 and context_hits >= 3


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _load_used() -> list:
    if os.path.exists(USED_TOPICS_FILE):
        try:
            with open(USED_TOPICS_FILE) as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _is_duplicate(title: str, used: list) -> bool:
    norm = _normalize(title)
    return any(_normalize(t) == norm for t in used)


# ---------------------------------------------------------------- output fields

def _verification_status(c: dict) -> str:
    subreddit = c["source"].split(":", 1)[-1]
    if subreddit == "badhistory":
        return "myth_debunked"
    if subreddit in CONTESTED_SUBREDDITS:
        return "needs_check"
    return "verified"


def _make_hook(c: dict) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", c["content"])
    first = next((s for s in sentences if len(s.split()) > 5), c["title"])
    return first[:100].rstrip(".") + "."


def _extractive_summary(content: str, words: int = 170) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", content)
    summary, total = [], 0
    for s in sentences:
        w = len(s.split())
        if total + w > words:
            break
        summary.append(s)
        total += w
    return " ".join(summary).strip()[:1000]


def _est_length_min(content: str) -> int:
    n = len(content)
    if n < 2000:
        return 3
    if n < 4000:
        return 4
    return 5


def _to_topic(c: dict) -> dict:
    source_name = c["source"].split(":", 1)[-1]
    category = c.get("category") or CATEGORY_MAP.get(source_name, "mystery")
    return {
        "title": c["title"],
        "hook": _make_hook(c),
        "category": category,
        "source_urls": [u for u in c["source_urls"] if u],
        "summary": _extractive_summary(c["content"]),
        "verification_status": _verification_status(c),
        # ranked by source score, scaled to 0-100
        "novelty_score": min(100, int(c["score"] / 500)),
        "est_video_length_min": _est_length_min(c["content"]),
        # internal plumbing fields for the downstream pipeline
        "source": c["source"],
        "content": c["content"],
        "score": c["score"],
        "upvote_ratio": c["upvote_ratio"],
    }


# ---------------------------------------------------------------- main

def run_topic_generation(limit: int = 100, target: int = TARGET_TOPICS, use_browser: bool = True) -> list:
    os.makedirs(TOPICS_OUTPUT_DIR, exist_ok=True)
    used = _load_used()

    raw = []

    # 1) Primary: Wikipedia (stable, not ISP-blocked)
    print(f"Pulling niche topics from Wikipedia...")
    raw.extend(fetch_wikipedia_candidates(limit=limit, target=target, used_titles=used))

    # 2) Secondary: Reddit (best-effort — unreliable on this ISP).
    #    Skipped unless REDDIT_ENABLED=1 (or --limit high forces it) to keep runs fast.
    reddit_raw = []
    if os.environ.get("REDDIT_ENABLED") == "1":
        token = _get_oauth_token()
        if token:
            print("  [reddit] Using Reddit OAuth API...")
            session = _session()
            for sub in REDDIT_SUBREDDITS:
                reddit_raw.extend(fetch_reddit_oauth(token, sub, limit))
                time.sleep(1)
        else:
            print("  [reddit] Using requests + browser cookie...")
            session = _session()
            for sub in REDDIT_SUBREDDITS:
                reddit_raw.extend(fetch_reddit(session, sub, limit))
                time.sleep(1)

        if not reddit_raw and use_browser and PLAYWRIGHT_AVAILABLE:
            print("  [reddit] Using headless Chromium + browser cookie...")
            reddit_raw = fetch_reddit_playwright(REDDIT_SUBREDDITS, limit)

    if reddit_raw:
        raw.extend(reddit_raw)
    else:
        print("  [reddit] skipped — Wikipedia only")

    if not raw:
        print("\n  ⚠️  No data from any source.\n")
        return []

    print(f"\nTotal raw candidates: {len(raw)}")

    passed = [c for c in raw if _passes_filters(c) and not _is_duplicate(c["title"], used)]
    print(f"After filters + dedupe: {len(passed)}")

    # Rank: story-driven Reddit subs first, otherwise by score. Wikipedia has a
    # flat score so order within it is stable.
    def _rank_key(c: dict):
        if c.get("source") == "wikipedia":
            return (0, 0, c["title"])
        return (0, 1, c["score"])

    passed.sort(key=_rank_key, reverse=True)
    topics = [_to_topic(c) for c in passed[:target]]
    print(f"Selected: {len(topics)} topics")

    _save_used(used + [t["title"] for t in topics])

    batch_file = os.path.join(
        TOPICS_OUTPUT_DIR, f"batch_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(batch_file, "w") as f:
        json.dump(topics, f, indent=2)
    print(f"Batch saved to {batch_file}")

    return topics


def _save_used(titles: list) -> None:
    with open(USED_TOPICS_FILE, "w") as f:
        json.dump(titles, f, indent=2)


def load_latest_topics() -> list:
    """Load the most recently generated topic batch from topics/."""
    if not os.path.isdir(TOPICS_OUTPUT_DIR):
        return []
    batches = sorted(
        (os.path.join(TOPICS_OUTPUT_DIR, f) for f in os.listdir(TOPICS_OUTPUT_DIR)
         if f.startswith("batch_") and f.endswith(".json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not batches:
        return []
    with open(batches[0]) as f:
        return json.load(f)