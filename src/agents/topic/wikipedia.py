"""Wikipedia source layer for the topic agent.

Wikipedia is the primary backstop source (stable, not ISP-blocked, API is
documented). ``fetch_wikipedia_candidates`` collects niche topic candidates
from search + category passes, returning them in the SAME dict shape as the
Reddit/LLM/channel paths so filtering/ranking/output stay shared.
"""
import re
import time
import urllib.parse

import requests

from src.agents.topic.helpers import _normalize

WIKI_API = "https://en.wikipedia.org/w/api.php"

# Wikipedia category -> topic category
# Story-driven only: WW1/WW2 battles & campaigns, human/animal experiments by
# the great powers, and dark legends/curses/hauntings. No weapon catalogs.
WIKI_CATEGORIES = {
    "Category:Battles of World War I": "story",
    "Category:Campaigns of World War I": "story",
    "Category:Battles and operations of World War II": "story",
    "Category:Campaigns of World War II": "story",
    "Category:People of World War II": "story",
    "Category:Japanese human subject research": "experiment",
    "Category:Nazi human subject research": "experiment",
    "Category:Human subject research": "experiment",
    "Category:Secret military programs": "experiment",
    "Category:Urban legends": "mystery",
    "Category:Curses": "mystery",
    "Category:Paranormal": "mystery",
    "Category:Last stands": "story",
}

# Search-driven pass (primary). These queries surface SPECIFIC story articles
# (e.g. "Unit 731", "Battle of Stalingrad", "Dybbøl legend") far better than
# the category lists. Each query is one HTTP request, so this stays light.
WIKI_SEARCH_QUERIES = [
    "human experimentation world war two", "Unit 731 experiments",
    "Nazi medical experiments prisoners", "chemical weapon test soldiers",
    "radiation experiments humans 1940s", "vivisection prisoners of war",
    "world war 2 battle last stand", "world war 1 trench raid",
    "world war 2 siege", "ww2 outnumbered battle", "ww1 epic battle story",
    "Battle of Stalingrad", "Battle of Britain pilot story",
    "WWII rescue mission escape story", "forgotten battle world war",
    "cursed object history", "haunted place true story",
    "urban legend true origin", "unsolved mystery cover-up",
    "vanished submarine mystery", "dark legend folklore curse",
]

# Generic concept/list pages that aren't specific stories. Substring blockers
# are kept tight on purpose: "massacre"/"war crime"/"world war" etc. are NOT
# here because specific dramatized pages ("Katyn massacre", "Babi Yar") are
# exactly what the niche wants — bare concept names are caught instead by the
# one-word blocks and the exact-title guard in _is_generic.
WIKI_GENERIC_BLOCKS = {
    "conspiracy theory", "human rights abuse",
    "biological warfare", "chemical warfare", "unexplained phenomenon",
    "list of", "disappeared person", "state terrorism", "political repression",
    "extrajudicial killing", "covert operation", "psychological warfare",
    "nuclear weapon", "history of", "human sexual activity",
    "human subject research", "human evolution", "animal testing",
    "human anatomy", "human body", "human genetics", "mental illness",
    "brainwashing", "mind control", "human", "human behavior",
    "classified information", "ghost hunting", "trump", "president",
    "presidential", "election", "politician", "political party",
    "enforced disappearance", "aircraft nuclear propulsion", "nuclear propulsion",
    "aircraft carrier", "propulsion", "weapons of mass destruction",
    "weapon of mass destruction",
}

# One-word generic titles that are never specific stories.
WIKI_ONE_WORD_BLOCKS = {
    "human", "brainwashing", "war", "army", "navy", "weapon", "bomb",
    "missile", "torture", "prison", "disease", "virus", "experiment",
    "science", "history", "mystery", "conspiracy", "myth", "legend",
    "soldier", "death", "massacre", "genocide", "prisoner", "disappearance",
}

# Bare concept titles that must never surface as topics (the specific-stories
# rule: a topic needs a named subject, not a concept).
WIKI_EXACT_TITLE_BLOCKS = {
    "world war", "world war i", "world war 1", "world war one",
    "world war ii", "world war 2", "world war two",
    "war crime", "war crimes", "massacre", "genocide", "mass killing",
    "human experiments", "human experimentation", "experiments on humans",
    "animal testing", "vivisection",
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
    "battle", "battled", "ambush", "siege", "invaded", "invasion", "raid",
    "assault", "airstrike", "air strike", "struck", "downed", "crashed",
    "surrounded", "outnumbered", "veteran", "combat", "firefight", "war",
    "armed conflict", "occupation", "insurgency", "guerrilla",
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


_wiki_last_request = 0.0
WIKI_MIN_INTERVAL = 1.2  # seconds between ANY Wikipedia request (rate limit ~1/s)


def _wiki_throttle():
    """Ensure a minimum gap between all Wikipedia requests."""
    global _wiki_last_request
    elapsed = time.monotonic() - _wiki_last_request
    if elapsed < WIKI_MIN_INTERVAL:
        time.sleep(WIKI_MIN_INTERVAL - elapsed)
    _wiki_last_request = time.monotonic()


def _wiki_get_json(params: dict) -> dict:
    params.setdefault("format", "json")
    for attempt in range(3):
        try:
            _wiki_throttle()
            r = requests.get(
                WIKI_API, params=params, headers=WIKI_HEADERS, timeout=20, verify=False
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                try:
                    wait = int(retry_after) if retry_after is not None else 0
                except (TypeError, ValueError):
                    wait = 0
                if wait <= 0:
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
    return results


def _is_generic(title: str) -> bool:
    """True for list/concept/media pages that aren't specific stories."""
    low = title.lower()
    if low in WIKI_EXACT_TITLE_BLOCKS:
        return True
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


def _fresh_titles(titles: list, seen: set, used: set) -> list:
    """Keep titles not seen/used; records the seen set in place."""
    fresh = []
    for t in titles:
        low = t.lower()
        if low in seen or low in used:
            seen.add(low)
            continue
        seen.add(low)
        fresh.append(t)
    return fresh


def _search_pass(target: int, seen: set, used: set, candidates: list) -> None:
    """Primary pass: page through each search query until we hit target."""
    for q in WIKI_SEARCH_QUERIES:
        for page in range(3):
            if len(candidates) >= target:
                return
            titles = _wiki_search(q, limit=10, offset=page * 10)
            if not titles:
                break
            fresh = _fresh_titles([t for t in titles if not _is_generic(t)], seen, used)
            if not fresh:
                continue
            extracts = _wiki_extract_many(fresh)
            for t in fresh:
                c = _candidate(t, "mystery", extracts)
                if c:
                    candidates.append(c)
                    if len(candidates) >= target:
                        return


def _category_pass(target: int, seen: set, used: set, candidates: list) -> None:
    """Enrichment pass: pull whole categories only if search came up short."""
    for cat, category in WIKI_CATEGORIES.items():
        if len(candidates) >= target:
            return
        titles = _wiki_category_members(cat, 40)
        print(f"    [wiki] {cat}: {len(titles)} pages")
        fresh = _fresh_titles(titles, seen, used)
        if not fresh:
            continue
        extracts = _wiki_extract_many(fresh)
        for t in fresh:
            c = _candidate(t, category, extracts)
            if c:
                candidates.append(c)
                if len(candidates) >= target:
                    return


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

    _search_pass(target, seen, used, candidates)
    if len(candidates) < target:
        _category_pass(target, seen, used, candidates)

    print(f"    [wiki] total: {len(candidates)} candidates")
    return candidates


def _verify_on_wikipedia(title: str) -> bool:
    """Best-effort: does this topic correspond to a real Wikipedia article?

    Searches with the topic's proper nouns (capitalized words like "Vela",
    "Byford Dolphin") plus year tokens. Accepts if the top search result
    shares a large fraction of the proposed title's tokens — a hallucinated
    subject gets a weak/empty match, a real one returns its article.
    On rate-limit/error we return True (don't block a run for throttling).
    """
    words = re.findall(r"[A-Za-z][a-z]+|\d{3,4}", title)
    key = [w for w in words if (w[0].isupper() and w.lower() not in ("the", "a", "an")) or (w.isdigit() and len(w) == 4)]
    if len(key) < 2:
        return True
    # Use the two most distinctive proper nouns; Wikipedia finds a real article
    # (e.g. "Vela incident") but not a hallucinated one.
    query = " ".join(key[:2])
    try:
        hits = _wiki_search(query, limit=5)
        if not hits:
            return False
        top = _normalize(hits[0])
        k1, k2 = _normalize(key[0]), _normalize(key[1])
        return k1 in top and k2 in top
    except Exception:
        return True


# ------------------------------------------------------------ source imagery
# The asset stage can only show what Commons actually hosts. A subject with
# no free imagery leaves the pipeline picking whatever matched a keyword, so
# a topic like that is rejected before any research time is spent on it.
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
MIN_SOURCE_IMAGES = 1   # non-chrome images on the resolved article
MIN_COMMONS_HITS = 3    # files Commons can offer for the resolved subject

# Chrome that sits on an article but can never be shown on screen.
_WIKI_ICONISH = re.compile(
    r"flag|icon|logo|symbol|map|locator|coat of arms|banner|seal|"
    r"padlock|disambig|arrow|question book|ambox|wiki|"
    r"audio|copyright|non-free|edit-|crystal",
    re.I,
)

# Words that carry no subject information in a story headline. A title is
# written as a sentence, so proper-noun matching alone would ask Wikipedia
# for "Zodiac 2020" and get nothing back.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "from", "with", "and",
    "or", "but", "who", "whom", "whose", "what", "when", "where", "why",
    "how", "that", "which", "this", "these", "those", "his", "her", "its",
    "their", "has", "have", "had", "was", "were", "are", "been", "being",
    "is", "be", "for", "by", "as", "into", "after", "before", "about",
    "over", "under", "than", "then", "them", "they", "you", "your", "one",
    "two", "new", "old", "most", "more", "some", "any", "all", "out", "up",
    "til", "did", "does", "really", "finally", "revealed", "story",
}


def _subject_tokens(title: str) -> list:
    """Words a headline is actually about, years pushed to the end.

    Short function words and filler are dropped, and years are appended last
    because AND-ing a year into a search query ("Zodiac killer cipher 2020")
    matches no article at all while "Zodiac killer cipher" does.
    """
    words = re.findall(r"[A-Za-z][a-z]+|\d{3,4}", title)
    named, eras = [], []
    for w in words:
        if w.isdigit():
            eras.append(w)
        elif len(w) >= 4 and w.lower() not in _STOPWORDS:
            named.append(w)
    return (named + eras)[:6]


def _resolve_article(title: str) -> tuple:
    """Search for the article this story is about.

    Returns (page title, non-chrome image titles). An empty title means the
    subject could not be resolved — either the API failed or no article
    matches those words. Callers must treat that as "unknown", never as
    "no imagery": story headlines routinely fail to match an article title.
    """
    key = _subject_tokens(title)
    if len(key) < 2:
        return "", []
    want = {_normalize(t) for t in key[:4]}

    # MediaWiki ANDs every term, so try three words and fall back to two.
    # Results are ranked by how much of the subject they actually contain —
    # a bare two-word query otherwise hands back a namesake for the real one.
    for query in (" ".join(key[:3]), " ".join(key[:2])):
        data = _wiki_get_json({
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 0,
            "gsrlimit": 5,
            "prop": "images",
            # NOTE: this limit is shared across the WHOLE response, not per
            # page — one large article would otherwise eat the budget and
            # starve the page we actually want.
            "imlimit": 500,
            "redirects": 1,
        })
        if not data:
            return "", []         # throttled / offline: unknown, not a miss
        pages = ((data.get("query") or {}).get("pages")) or {}
        best_title, best_images, best_rank = "", [], (-1, -1)
        for page in pages.values():
            if page.get("missing"):
                continue
            overlap = len(want & set(_normalize(page.get("title") or "").split()))
            if overlap < 2:
                continue          # unrelated article
            images = [
                i.get("title", "")
                for i in (page.get("images") or [])
                if i.get("title") and not _WIKI_ICONISH.search(i["title"])
            ]
            rank = (overlap, len(images))   # right subject first, richer next
            if rank <= best_rank:
                continue
            best_title, best_images, best_rank = (
                page.get("title", ""), images, rank
            )
        if best_title:
            return best_title, best_images
    return "", []


def _commons_hits(subject: str):
    """Bitmap files Commons offers for `subject`, or None when it could not ask."""
    if not subject:
        return 0
    try:
        _wiki_throttle()
        r = requests.get(
            COMMONS_API,
            params={
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": f"filetype:bitmap {subject}",
                "srnamespace": 6,
                "srlimit": 10,
            },
            headers=WIKI_HEADERS,
            timeout=20,
            verify=False,
        )
        if r.status_code != 200:
            return None
        return int(((r.json().get("query") or {}).get("searchinfo") or {})
                   .get("totalhits", 0))
    except (requests.RequestException, ValueError):
        return None


def has_source_imagery(title: str, cache: dict = None) -> bool:
    """True when free imagery for this subject plausibly exists.

    Two checks, because they fail in different ways:
      * the article must carry at least one non-chrome image, proving the
        subject itself is pictured at all;
      * Commons must actually host files for it — a Wikipedia press photo is
        often non-free, which is exactly how modern subjects end up with
        nothing correct to show.

    A subject that does not resolve to an article, a rate limit, or a network
    failure all return True: this gate only blocks topics it can positively
    prove are unsourceable, because a story headline that fails to match an
    article title says nothing about the imagery.
    """
    if cache is not None and title in cache:
        return cache[title]
    try:
        page_title, images = _resolve_article(title)
        if not page_title:
            result = True         # could not judge
        elif len(images) < MIN_SOURCE_IMAGES:
            result = False        # article carries only chrome
        else:
            hits = _commons_hits(page_title)
            result = True if hits is None else hits >= MIN_COMMONS_HITS
    except Exception as e:
        print(f"    [imagery] {title[:40]!r}: {e} — not blocking")
        result = True
    if cache is not None:
        cache[title] = result
    return result
