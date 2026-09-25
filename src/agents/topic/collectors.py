"""Collectors: one function per source kind, all returning list[Lead].

Design rules:
  * A broken source must never kill the agent — every collector is guarded and
    returns [] on failure; the agent logs the count.
  * Full article bodies are only fetched for index sources that ask for it
    (``fetch_articles > 0``), and only for the newest few links, via a small
    thread pool. RSS/Atom entries already carry their summary.
  * All HTTP goes through topic_agent.http so caching and the polite UA are
    uniform. (module lives at src.agents.topic.http)
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from requests import RequestException

from src.agents.topic.http import fetch, fetch_json
from src.agents.topic.leads import Lead, clean_text, first_paragraph
from src.agents.topic.sources import spec_ttl_hours

_BODY_CAP_CHARS = 6000
_MAX_LINKS_PER_PAGE = 30
_MAX_QUERIES = 4

# Boilerplate that leaks into article bodies from template-heavy sites:
# image-caption chrome, edit/author footers, donation appeals, glossaries.
_BOILERPLATE_PATTERNS = (
    (re.compile(r"More information about this image", re.I), ""),
    (re.compile(r"Close Image Lightbox", re.I), ""),
    (re.compile(
        r"Last Edited:.*?View the list of all donors", re.S | re.I
    ), ""),
    (re.compile(
        r"Feedback.*?View the list of all donors", re.S | re.I
    ), ""),
    (re.compile(
        r"Glossary\s+Full Glossary\s+Close glossary", re.I
    ), ""),
    (re.compile(r"\bAuthor\(s\):\s*[A-Z][A-Za-z ,.]*", re.I), ""),
)


def strip_boilerplate(text: str) -> str:
    """Remove site-level chrome (captions, footers, donor/glossary blocks)."""
    if not text:
        return text
    for pattern, repl in _BOILERPLATE_PATTERNS:
        text = pattern.sub(repl, text)
    return clean_text(text)


def extract_article_body(url: str, selectors: list[str] | None = None) -> str:
    """Fetch one URL and return its cleaned article body ("" if unusable).

    Public entry point for pulling a full article body from a plain URL —
    used where a lead/rss summary is too thin to script from (e.g. story
    research at script time).
    """
    try:
        html = fetch(url, timeout=20)
    except RequestException:
        return ""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return _extract_body(soup, selectors or _GENERIC_BODY_SELECTORS)


def collect(spec: dict, limit: int = 200) -> list[Lead]:
    """Run the collector matching spec['kind']. Never raises."""
    kind = spec.get("kind", "index")
    try:
        if kind == "rss":
            leads = _collect_rss(spec)
        elif kind == "index":
            leads = _collect_index(spec)
        elif kind == "api":
            leads = _collect_api(spec)
        elif kind == "mediawiki":
            leads = _collect_mediawiki(spec)
        else:
            leads = []
        return leads[:limit]
    except Exception:
        return []


# ------------------------------------------------------------------ rss/atom

def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _parse_rss_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    try:
        return parsedate_to_datetime(raw)
    except (ValueError, TypeError):
        pass
    iso = raw.rstrip("Z")
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


def _feed_entry_text(node) -> str:
    """Best available body/summary text from an RSS item or Atom entry."""
    for tag in ("content:encoded", "encoded", "content", "summary", "description"):
        el = node.find(tag)
        if el is not None and el.text.strip():
            soup = BeautifulSoup(el.text, "html.parser")
            return clean_text(soup.get_text(" "))
    return ""


def _feed_link(node, local: str) -> str | None:
    if local == "entry":
        for link in node.find_all("link"):
            rel = link.get("rel")
            if rel in (None, "alternate"):
                href = link.get("href")
                if href:
                    return href
        return None
    link = node.find("link")
    if link is not None:
        if link.get("href"):
            return link.get("href")
        return link.text.strip()
    return None


def _collect_rss(spec: dict) -> list[Lead]:
    leads: list[Lead] = []
    ttl = spec_ttl_hours(spec)
    for feed_url in spec["starts"][:_MAX_QUERIES]:
        try:
            body = fetch(feed_url, ttl=ttl)
        except RequestException:
            continue
        root = BeautifulSoup(body, "xml").find()
        if root is None:
            continue
        for node in root.find_all():
            if _local_name(node.name) not in ("item", "entry"):
                continue
            title = node.find("title")
            title = clean_text(title.get_text(" ")) if title is not None else ""
            url = _feed_link(node, _local_name(node.name))
            if not title or not url:
                continue
            raw = _feed_entry_text(node)
            published = (
                _parse_rss_date(node.find("pubdate").text)
                if (node.find("pubdate") is not None)
                else _parse_rss_date(
                    _first_text(node, ("published", "updated", "date"))
                )
            )
            leads.append(
                Lead(
                    title=title,
                    url=urljoin(feed_url, url),
                    source_key=spec["key"],
                    source_name=spec["name"],
                    niche=spec["niche"],
                    published=published,
                    summary=first_paragraph(raw),
                    body=raw,
                )
            )
    return leads


def _first_text(node, names: tuple) -> str | None:
    for name in names:
        el = node.find(name, recursive=False) or node.find(name)
        if el is not None and el.text.strip():
            return el.text.strip()
    return None


# ------------------------------------------------------------------ index

_DATE_META = (
    ("meta", {"property": "article:published_time"}),
    ("meta", {"property": "og:article:published_time"}),
    ("meta", {"name": "date"}),
)

_GENERIC_BODY_SELECTORS = (
    "main",
    "article",
    ".entry-content",
    ".post-content",
    ".article-body",
    ".td-post-content",
    ".field--name-body",
    ".content",
)


def _extract_body(soup: BeautifulSoup, selectors: list[str]) -> str:
    candidates: list = []
    for sel in selectors:
        for el in soup.select(sel):
            candidates.append(el)
    if not candidates:
        for sel in _GENERIC_BODY_SELECTORS:
            for el in soup.select(sel):
                candidates.append(el)
    if not candidates:
        return ""
    best = max(candidates, key=lambda el: len(el.get_text(" ", strip=True)))
    for tag in best.find_all(["script", "style", "nav", "aside", "form", "iframe"]):
        tag.decompose()
    return strip_boilerplate(best.get_text(" ", strip=True))[:_BODY_CAP_CHARS]


def _meta_date(soup: BeautifulSoup) -> datetime | None:
    for name, attrs in _DATE_META:
        el = soup.find(name, attrs=attrs)
        if el is not None:
            return _parse_rss_date(el.get("content") or el.text)
    time_el = soup.find("time")
    if time_el is not None and time_el.get("datetime"):
        return _parse_rss_date(time_el.get("datetime"))
    return None


def _harvest_links(soup: BeautifulSoup, spec: dict, base: str) -> list[dict]:
    """Return [{'title','url'}] from <a> tags matching link_match/exclude."""
    matches = spec["link_match"] or []
    excludes = spec["exclude"] or []
    found: list[dict] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.startswith("mailto:"):
            continue
        url = urljoin(base, href)
        if matches and not any(m in url for m in matches):
            continue
        if any(x in url for x in excludes):
            continue
        if url in seen:
            continue
        seen.add(url)
        title = clean_text(a.get_text(" "))
        if not title or len(title) < 4:
            alt = a.find("img")
            title = clean_text(alt.get("alt")) if alt is not None else ""
        if not title:
            continue
        found.append({"title": title, "url": url})
        if len(found) >= _MAX_LINKS_PER_PAGE:
            break
    return found


def _collect_index(spec: dict) -> list[Lead]:
    leads: list[Lead] = []
    ttl = spec_ttl_hours(spec)
    candidates: list[dict] = []

    for start in spec["starts"][:_MAX_QUERIES]:
        try:
            html = fetch(start, ttl=ttl)
        except RequestException:
            continue
        soup = BeautifulSoup(html, "html.parser")
        candidates.extend(_harvest_links(soup, spec, start))
        paginate = spec.get("paginate")
        if paginate:
            for n in range(2, 1 + spec.get("max_pages", 1)):
                page_url = urljoin(start, paginate.format(n=n))
                try:
                    page_html = fetch(page_url, ttl=ttl)
                except RequestException:
                    continue
                page_soup = BeautifulSoup(page_html, "html.parser")
                candidates.extend(_harvest_links(page_soup, spec, page_url))

    for cand in candidates[:_MAX_LINKS_PER_PAGE]:
        leads.append(
            Lead(
                title=cand["title"],
                url=cand["url"],
                source_key=spec["key"],
                source_name=spec["name"],
                niche=spec["niche"],
            )
        )

    fetch_n = spec.get("fetch_articles", 0)
    if fetch_n and leads:
        leads = _fetch_bodies(leads, spec, fetch_n)
    return leads


def _fetch_bodies(leads: list[Lead], spec: dict, n: int) -> list[Lead]:
    """Fetch full text for the newest `n` leads of a spec, in parallel."""
    ttl = spec_ttl_hours(spec)
    selectors = spec.get("body_selectors") or []
    lock = threading.Lock()
    done = 0

    def work(lead: Lead):
        nonlocal done
        try:
            html = fetch(lead.url, ttl=ttl, timeout=25)
        except RequestException:
            return
        soup = BeautifulSoup(html, "html.parser")
        body = _extract_body(soup, selectors)
        if not body:
            return
        with lock:
            lead.body = body
            lead.summary = first_paragraph(body)
            if lead.published is None:
                lead.published = _meta_date(soup)
            done += 1

    pool = ThreadPoolExecutor(max_workers=4)
    futures = [pool.submit(work, lead) for lead in leads[:n]]
    for _ in as_completed(futures, timeout=120):
        pass
    pool.shutdown(wait=True)
    return leads


# ------------------------------------------------------------------ api

def _deep_get(item, path: str):
    """Dotted-path lookup; a list in the way uses its first element."""
    cur = item
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[0] if cur else None
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, list):
        cur = cur[0] if cur else None
    if isinstance(cur, dict):
        for key in ("text", "value", "title", "@value"):
            if key in cur:
                return cur[key]
    return cur


def _map_api_item(spec: dict, item: dict):
    mapping = spec.get("api_map") or {}
    title_path = mapping.get("title", "title")
    summary_path = mapping.get("summary", "")
    date_path = mapping.get("date", "")
    url_path = mapping.get("url", "url")

    title = _deep_get(item, title_path)
    if not isinstance(title, str) or not title.strip():
        return None
    title = clean_text(title)[:300]

    if isinstance(url_path, tuple):
        field, template = url_path
        key = _deep_get(item, field)
        url = template.format(**{field: key}) if key is not None else ""
    else:
        url = _deep_get(item, url_path) or ""

    raw_date = _deep_get(item, date_path) if date_path else None
    published = _parse_rss_date(raw_date) if isinstance(raw_date, str) else None
    summary = _deep_get(item, summary_path) if summary_path else ""
    if isinstance(summary, list):
        summary = " ".join(str(s) for s in summary)
    if not isinstance(summary, str):
        summary = ""

    return {
        "title": title,
        "url": url,
        "summary": first_paragraph(summary),
        "published": published,
    }


def _collect_api(spec: dict) -> list[Lead]:
    leads: list[Lead] = []
    ttl = spec_ttl_hours(spec)
    base = spec["starts"][0]
    for query in spec.get("api_params", [])[:_MAX_QUERIES]:
        try:
            data = fetch_json(base, ttl=ttl, timeout=30, params=dict(query))
        except RequestException:
            continue
        items = data
        for part in (spec.get("api_items") or "").split("."):
            if not isinstance(items, dict):
                items = None
                break
            items = items.get(part)
        if not isinstance(items, list):
            continue
        for it in items[: spec.get("api_rows", 25)]:
            if not isinstance(it, dict):
                continue
            mapped = _map_api_item(spec, it)
            if mapped is None:
                continue
            leads.append(
                Lead(
                    title=mapped["title"],
                    url=mapped["url"],
                    source_key=spec["key"],
                    source_name=spec["name"],
                    niche=spec["niche"],
                    published=mapped["published"],
                    summary=mapped["summary"],
                )
            )
    return leads


# ------------------------------------------------------------------ mediawiki

_MW_IGNORED_NS = ("File:", "Template:", "Category:", "Wikipedia:", "Help:",
                  "Talk:", "Special:", "Portal:", "User:", "MediaWiki:")


def _collect_mediawiki(spec: dict) -> list[Lead]:
    """Browse Wikipedia category pages (HTML #mw-pages pane) for article links.

    The MediaWiki categorymembers API returned empty results for these
    categories during the build, while the rendered category page lists them
    fine — so we harvest the HTML directly.
    """
    leads: list[Lead] = []
    ttl = spec_ttl_hours(spec)
    for category in spec["starts"][:4]:
        title = category if category.startswith("Category:") else f"Category:{category}"
        try:
            html = fetch(f"https://en.wikipedia.org/wiki/{title}", ttl=ttl, timeout=30)
        except RequestException:
            continue
        soup = BeautifulSoup(html, "html.parser")
        pane = soup.select_one("div#mw-pages")
        if pane is None:
            continue
        seen: set[str] = set()
        for a in pane.select("a[href^='/wiki/']"):
            article = a.get("href", "")[6:]
            if any(article.startswith(ns) for ns in _MW_IGNORED_NS):
                continue
            name = a.get_text(" ", strip=True)
            if not name or len(name) < 4 or name in seen:
                continue
            seen.add(name)
            leads.append(
                Lead(
                    title=name,
                    url=f"https://en.wikipedia.org/wiki/{article}",
                    source_key=spec["key"],
                    source_name=spec["name"],
                    niche=spec["niche"],
                )
            )
            if len(leads) >= 100:
                return leads
    return leads