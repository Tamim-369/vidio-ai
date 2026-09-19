"""War-story topic sources.

Pulls fresh, story-driven war material from three independent places:

1. Live conflict news (Google News RSS) — surfaces the CURRENT wars as they
   unfold (the 2026 US–Iran war, Houthi/Red Sea escalation, etc.).
2. War-history websites (RSS feeds + curated pages) — long-form, story-driven
   military history pieces (Vietnam / Iraq / Cold War / weird experiments).
3. Wikipedia war categories / search — durable backstop for named battles,
   operations, incidents, and experiments.

Every function returns the SAME dict shape as research_pipeline's Wikipedia
source: {"title", "url", "content"}. That lets the existing LLM idea-extraction,
vetting, dedup, and ranking layers consume this untouched.
"""
import re
import time
import urllib.parse
import urllib.request
import ssl

from src.services.topic_generator import (
    _wiki_category_members,
    _wiki_search,
    _wiki_extract_many,
)

# --------------------------------------------------------------------------- news
# Google News RSS is query-based and unrestricted (no API key). Each query is
# a separate request, so keep the list tight.
NEWS_QUERIES = [
    '"US Iran war" Iran',
    'Iran Houthi Red Sea Yemen war',
    'US Iran naval blockade Hormuz',
    'Iran drone missile attack Saudi',
    'US garrison attacked Middle East 2026',
]

# Wayback-independent, plain <item> RSS feeds for war-history magazines.
HISTORY_FEEDS = [
    "https://militaryhistorynow.com/feed/",
    "https://www.warhistoryonline.com/feed",
]

# Curated war-story pages (first-hand archives with real narratives).
STORY_URLS = [
    ("vnwarstories.com — The War Dance (1967)", "http://vnwarstories.com/index.html"),
    ("West Point oral history — Vietnam 173rd", "https://westpointcoh.org/interviews/one-grunt-s-weird-tour-19-and-a-half-months-in-vietnam-with-the-173rd"),
]

WAR_STORY_KEYWORDS = [
    "battle", "siege", "ambush", "raid", "operation", "invasion", "campaign",
    "war", "soldier", "veteran", "marines", "air force", "navy", "squadron",
    "fighter jet", "tank", "submarine", "helicopter", "drone", "missile",
    "nuclear", "experiment", "secret", "Von", "Lost", "hunt", "fight",
    "crash", "disaster", "radiation", "espionage", "spy", "agent",
]

_CTX = ssl._create_unverified_context()
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}


def _http_get(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
        return r.read().decode("utf-8", "ignore")


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_feed_items(url: str, limit: int = 30) -> list:
    """Parse <item> entries from an RSS/Atom feed URL."""
    try:
        body = _http_get(url)
    except Exception as e:
        print(f"    [war:feed] {url} failed ({str(e)[:60]})")
        return []
    items = re.findall(r"<item>(.*?)</item>", body, re.S)
    parsed = []
    for it in items[:limit]:
        t = re.search(r"<title>(.*?)</title>", it, re.S)
        l = re.search(r"<link>(.*?)</link>", it, re.S)
        d = re.search(r"<description>(.*?)</description>", it, re.S)
        ct = re.search(r"<content:encoded>(.*?)</content:encoded>", it, re.S)
        title = _strip_html(t.group(1)) if t else ""
        link = _strip_html(l.group(1)) if l else ""
        desc = _strip_html(ct.group(1) if ct else (d.group(1) if d else ""))
        if title and link:
            parsed.append({"title": title, "url": link, "content": desc})
    return parsed


def fetch_news_sources(limit: int = 40) -> list:
    """Google News RSS for current conflict stories. Returns source dicts."""
    sources = []
    seen = set()
    for q in NEWS_QUERIES:
        url = (
            "https://news.google.com/rss/search?q="
            + urllib.parse.quote(q) + "&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            body = _http_get(url)
        except Exception as e:
            print(f"    [war:news] {q} failed ({str(e)[:60]})")
            time.sleep(1)
            continue
        for it in re.findall(r"<item>(.*?)</item>", body, re.S)[:10]:
            t = re.search(r"<title>(.*?)</title>", it, re.S)
            l = re.search(r"<link>(.*?)</link>", it, re.S)
            d = re.search(r"<description>(.*?)</description>", it, re.S)
            title = _strip_html(t.group(1)) if t else ""
            link = _strip_html(l.group(1)) if l else ""
            desc = _strip_html(d.group(1)) if d else ""
            key = title.lower()
            if title and link and key not in seen and len(desc) > 80:
                seen.add(key)
                sources.append({
                    "title": title,
                    "url": link if link.startswith("http") else f"https://news.google.com{link}",
                    "content": f"{title}.\n{desc}",
                })
        time.sleep(0.5)
        if len(sources) >= limit:
            break
    print(f"  [war:news] {len(sources)} live conflict items")
    return sources[:limit]


def fetch_history_sources(limit: int = 30) -> list:
    """RSS feeds from war-history magazines (long-form story material)."""
    sources = []
    seen = set()
    for url in HISTORY_FEEDS:
        for item in _fetch_feed_items(url, limit=20):
            low = item["title"].lower()
            if not any(k.lower() in low for k in WAR_STORY_KEYWORDS):
                continue
            key = item["title"].lower()
            if key in seen or len(item["content"]) < 200:
                continue
            seen.add(key)
            sources.append(item)
    print(f"  [war:sites] {len(sources)} history-magazine stories")
    return sources[:limit]


def _extract_story_from_html(html: str, title: str) -> str:
    """Pull the readable narrative text out of a static story page."""
    # West Point oral history pages wrap the interview in <p>; vnwarstories
    # uses <td> blocks. Grab the longest run of text either way.
    text = _strip_html(html)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Clip to the first ~60 sentences of actual body content.
    body = " ".join(s for s in sentences if len(s) > 40)[:6000]
    return body or title


def fetch_curated_sources() -> list:
    """Hand-picked first-hand war-story pages (stable, high quality)."""
    sources = []
    for title, url in STORY_URLS:
        try:
            html = _http_get(url)
            content = _extract_story_from_html(html, title)
            if len(content) > 300:
                sources.append({"title": title, "url": url, "content": content})
                print(f"  [war:curated] +1 ({title[:50]})")
        except Exception as e:
            print(f"    [war:curated] {title[:40]} failed ({str(e)[:60]})")
    return sources


def fetch_war_wikipedia(categories: list, queries: list, limit: int = 30) -> list:
    """Wikipedia categories + search -> war-story article sources."""
    titles = []
    seen = set()
    for cat in categories:
        try:
            for t in _wiki_category_members(cat, limit=30):
                if t.lower() not in seen:
                    seen.add(t.lower())
                    titles.append(t)
        except Exception:
            continue
    for q in queries:
        try:
            for t in _wiki_search(q, limit=10):
                if t.lower() not in seen:
                    seen.add(t.lower())
                    titles.append(t)
        except Exception:
            continue

    extracts = _wiki_extract_many(titles[:limit])
    sources = []
    for t in titles[:limit]:
        ext = (extracts.get(t) or "").strip()
        if len(ext) < 300:
            continue
        # Wikipedia articles need a story/experiment signal, not a catalog.
        low = ext.lower()
        if not any(k in low for k in (
            "war", "battle", "ambush", "siege", "experiment", "secret",
            "attack", "raided", "air strike", "missile", "drone", "soldier",
            "troops", "invasion", "occupied", "killed",
        )):
            continue
        sources.append({
            "title": t,
            "url": f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}",
            "content": ext,
        })
    print(f"  [war:wiki] {len(sources)} articles")
    return sources


def fetch_all_war_sources(limit: int = 40) -> list:
    """Combined source pool: live news + history sites + curated pages.

    Wikipedia war articles flow through their own niche entries in the
    research pipeline (see NICHES), so this focuses on the non-Wikipedia live
    layer. Same {title, url, content} shape as the Wikipedia path.
    """
    merged = []
    merged += fetch_news_sources(limit=limit)
    merged += fetch_history_sources(limit=limit)
    merged += fetch_curated_sources()
    return merged