"""The verified source inventory (from the internet-research pass).

Every entry mirrors what was actually fetched and confirmed: real indexes,
RSS/Atom feeds, and documented JSON APIs with real query patterns. Selectors
are best-effort and the collectors degrade gracefully if a site changes — a
broken selector yields no leads, never a crash.

Spec kinds:
  rss        Atom/RSS feed; leads come from entries (title/link/summary).
  index      HTML list pages (category/tag/archive); leads come from <a> tags,
             optionally followed by a full-text fetch of the article body.
  api        JSON endpoint with typed response mapping (archive.org, loc.gov,
             Wellcome).
  mediawiki  MediaWiki API (categorymembers) or a Wikipedia page whose links
             are harvested (Wikipedia:Unusual_articles).
"""
from __future__ import annotations

SOURCES = [
    # ------------------------------------------------- ww1_ww2_stories
    {
        "key": "mhn",
        "name": "Military History Now",
        "kind": "rss",
        "niche": "ww1_ww2_stories",
        "starts": ["https://militaryhistorynow.com/feed"],
        "link_match": [],
        "exclude": [],
        "paginate": None,
        "max_pages": 1,
        "fetch_articles": 0,
        "body_selectors": [".entry-content", "article"],
    },
    {
        "key": "historynet",
        "name": "HistoryNet",
        "kind": "rss",
        "niche": "ww1_ww2_stories",
        "starts": ["https://historynet.com/feed"],
        "link_match": [],
        "exclude": [],
        "paginate": None,
        "max_pages": 1,
        "fetch_articles": 0,
        "body_selectors": [".entry-content", "article"],
    },
    {
        "key": "whonline",
        "name": "War History Online",
        "kind": "index",
        "niche": "ww1_ww2_stories",
        "starts": ["https://www.warhistoryonline.com/category/history/"],
        "link_match": ["warhistoryonline.com/"],
        "exclude": ["/category/", "/about", "/contact", "/legal", "/privacy", "/terms", "#"],
        "paginate": "page/{n}/",
        "max_pages": 2,
        "fetch_articles": 3,
        "body_selectors": [".entry-content", ".wpb_text_column", "article"],
    },
    {
        "key": "cmh",
        "name": "U.S. Army Center of Military History",
        "kind": "index",
        "niche": "ww1_ww2_stories",
        "starts": ["https://history.army.mil/Publications/Publications-Catalog/"],
        "link_match": ["/Publications/Publications-Catalog/"],
        "exclude": ["#"],
        "paginate": None,
        "max_pages": 1,
        "fetch_articles": 0,
        "body_selectors": [".content", "article"],
    },
    {
        "key": "wiki_wws",
        "name": "Wikipedia: WW1/WW2 battle categories",
        "kind": "mediawiki",
        "niche": "ww1_ww2_stories",
        "starts": [
            "Category:Battles_of_World_War_I",
            "Category:Battles_and_operations_of_World_War_II",
            "Category:Last stands",
            "Category:Campaigns_of_World_War_II",
        ],
        "link_match": [],
        "exclude": [],
        "paginate": None,
        "max_pages": 1,
        "fetch_articles": 0,
        "body_selectors": [],
    },
    # ------------------------------------------------------ experiments
    {
        "key": "nuremberg",
        "name": "Nuremberg Trials Project (HLS)",
        "kind": "index",
        "niche": "experiments",
        "starts": [
            "https://nuremberg.law.harvard.edu/transcripts/1?seq=1",
            "https://nuremberg.law.harvard.edu/transcripts/2?seq=1",
        ],
        "link_match": ["/"],
        "exclude": [
            "#", "/search/advanced", "/search/help", "/documents", "/trials",
            "/people", "/help", "/about", "/guide", "/funding",
        ],
        "paginate": None,
        "max_pages": 1,
        "fetch_articles": 0,
        "body_selectors": [".trial-detail", ".panel-body", "main"],
    },
    {
        "key": "ushmm",
        "name": "USHMM Holocaust Encyclopedia: Nazi medicine",
        "kind": "index",
        "niche": "experiments",
        "starts": [
            "https://encyclopedia.ushmm.org/content/en/article/nazi-medical-experiments",
        ],
        "link_match": ["?series="],
        "exclude": [
            "/ar/", "/cs/", "/de/", "/es/", "/fr/", "/he/", "/hu/", "/it/",
            "/ja/", "/ko/", "/nl/", "/pl/", "/pt/", "/ro/", "/ru/", "/sr/",
            "/tr/", "/uk/", "/zh/", "/content/en/other-", "/content/en/timeline",
            "/content/en/map", "/content/en/question", "/content/en/content",
        ],
        "paginate": None,
        "max_pages": 1,
        "fetch_articles": 2,
        "body_selectors": [".entry-content", ".article-content", "main"],
    },
    {
        "key": "wiki_cats",
        "name": "Wikipedia: experiment categories",
        "kind": "mediawiki",
        "niche": "experiments",
        "starts": [
            "Category:Human subject research",
            "Category:Nazi human subject research",
            "Category:Japanese human subject research",
            "Category:Biological warfare",
        ],
        "link_match": [],
        "exclude": [],
        "paginate": None,
        "max_pages": 1,
        "fetch_articles": 0,
        "body_selectors": [],
    },
    {
        "key": "wiki_dark",
        "name": "Wikipedia: dark-history categories",
        "kind": "mediawiki",
        "niche": "dark_legends",
        "starts": [
            "Category:Japanese war crimes",
            "Category:German war crimes",
            "Category:Nuremberg trials",
            "Category:Medicine in Nazi Germany",
        ],
        "link_match": [],
        "exclude": [],
        "paginate": None,
        "max_pages": 1,
        "fetch_articles": 0,
        "body_selectors": [],
    },
    {
        "key": "wiki_legends",
        "name": "Wikipedia: legend/horror categories",
        "kind": "mediawiki",
        "niche": "dark_legends",
        "starts": [
            "Category:Urban legends",
            "Category:Folklore",
            "Category:Curses",
            "Category:Reportedly haunted locations",
            "Category:Conspiracy theories",
        ],
        "link_match": [],
        "exclude": [],
        "paginate": None,
        "max_pages": 1,
        "fetch_articles": 0,
        "body_selectors": [],
    },
]

# Per-source authority used by the scorer (0..1). Primary sources and major
# analysts rank highest; listicle aggregators rank lowest.
AUTHORITY = {
    "cmh": 1.0,
    "mhn": 0.85,
    "historynet": 0.85,
    "whonline": 0.8,
    "wiki_wws": 0.7,
    "nuremberg": 1.0,
    "ushmm": 0.95,
    "wiki_cats": 0.7,
    "wiki_dark": 0.7,
    "wiki_legends": 0.7,
}

TTL_HOURS = {
    "rss": 6,
    "index": 6,
    "api": 12,
    "mediawiki": 24,
}


def spec_ttl_hours(spec: dict) -> float:
    return TTL_HOURS.get(spec.get("kind", "index"), 6) * 3600
