"""Deterministic helpers shared across the topic agent's generators.

Everything in here is pure logic / static data — no LLM, no network beyond
filesystem access. The two collection paths (the blank-prompt generator in
``generator.py`` and the sourced agent in ``agent.py``), the research agent,
and the pipeline all import from this single store so dedupe, used-vs-made
tracking, and niche-filtering behave identically everywhere.
"""
import json
import os
import re
import threading

from src.utils.file_helpers import OUTPUT_DIR

# Story subreddits, tiered by how well the sub's material maps to the
# channel's proven topic shape: a NAMED subject, physical stakes, and a
# truth-vs-lies framing, in 20-38s. Reddit exposes `score` (upvotes) but no
# view count, so popularity is enforced with a per-sub score floor below.
REDDIT_PRIMARY_SUBS = [
    "history",              # general history stories; the channel's best AVD
    "UnresolvedMysteries",  # unresolved stories with a subject and a stake
    "TrueCrime",            # historical crime, the channel's top performer
]
REDDIT_SECONDARY_SUBS = [
    "AncientHistory",  # 1 video @ 107.8% AVD (low volume, proven shape)
    "medievalhistory", # discussion-poor -> only its highest-scored posts
]
REDDIT_SUBREDDITS = REDDIT_PRIMARY_SUBS + REDDIT_SECONDARY_SUBS

# Subs whose content is inherently contested -> flag for manual check
CONTESTED_SUBREDDITS = set()

# Per-sub popularity floor. `forgottenhistory`-class subs top out in the
# tens of votes, while the story-shaped posts we want on the primary subs
# start in the thousands — one shared floor would either accept noise or
# reject the whole of a small sub.
REDDIT_SCORE_FLOORS = {
    "history": 3000,
    "UnresolvedMysteries": 3000,
    "TrueCrime": 2000,
    "AncientHistory": 800,
    "medievalhistory": 800,
}

# Crime / serial-killer / missing-person content to exclude from the niche.
# "was killed" / "death of" are intentionally ABSENT — casualty and destruction
# language is the substance of war-story topics.
CRIME_BLOCK_KEYWORDS = [
    "serial killer", "serial killer", "murder", "killer", "rapist", "stalker",
    "cctv", "kidnap", "abduction", "missing person", "body found",
    "school shooting", "mass shooter", "investigation", "arrested",
    "police", "criminal", "manslaughter",
]

# Subs where that same vocabulary IS the story rather than a mis-click, so
# the crime gate is suspended for them. The rest of the filters (meta
# patterns, score floor, upvote ratio, niche gate) still apply.
CRIME_EXEMPT_SUBS = {"TrueCrime", "UnresolvedMysteries"}

# The sub itself defines the niche, so no story keyword must be found in the
# title (the WW2/legend wording in NICHE_STRONG would otherwise reject every
# frontier/true-crime/ancient title out of hand).
TRUSTED_SUBS = {
    "history", "TrueCrime", "UnresolvedMysteries",
    "AncientHistory", "medievalhistory",
}

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
    # meta / engagement-farming shapes common on the story subs
    "discussion thread", "weekly discussion", "reminder that", "reminder:",
    "self-promo", "self promo", "does anyone else", "do you think",
    "am i wrong", "unpopular opinion", "change my mind", "would you",
    "what would happen if", "rate this", "how old is this", "identify this",
    "everyone's favourite", "everyone's favorite", "what's everyone's",
    "what is everyone's", "your favourite", "your favorite",
    "what are your thoughts", "your thoughts", "thoughts on this",
    "what do you think", "does anyone know", "does anyone remember",
    "has anyone seen", "anyone else", "i found this",
]

# Subs that reliably produce story-driven material -> slight ranking boost
STORY_SUBS = {
    "history", "TrueCrime", "UnresolvedMysteries",
    "AncientHistory", "medievalhistory",
}

CATEGORY_MAP = {
    "history": "story",
    "TrueCrime": "story",
    "UnresolvedMysteries": "mystery",
    "AncientHistory": "story",
    "medievalhistory": "story",
}

COOKIE_FILE = "cookie"
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

USED_TOPICS_FILE = "used_topics.json"
TOPICS_OUTPUT_DIR = "topics"
TARGET_TOPICS = 20

# Concurrent batch workers all call record_made_video when their video finishes;
# the read-modify-write must be serialized or a finished topic can be lost.
_used_lock = threading.Lock()

# Filter thresholds (from spec)
MIN_SCORE = 500
MIN_SELFTEXT = 1500
MIN_UPRATIO = 0.85

# Subs that are mostly images/links (no selftext) -> lower selftext floor + lower score floor
TITLE_BASED_SUBS = {
    "CombatFootage",
}

# Keywords that ALONE prove the topic is on-niche: secret human/animal
# experiments, WW1/WW2 battle stories, dark legends/curses/hauntings.
NICHE_STRONG = [
    "mystery", "mysterious", "secret", "top secret", "classified project",
    "unknown", "unexplained",
    "vanished", "disappeared", "disappear", "lost", "abandoned", "hidden",
    "curse", "cursed", "haunted", "creepy", "strange", "weird", "bizarre",
    "unsettling", "disturbing", "terrifying", "horrifying", "horror", "dark",
    "brutal", "torture", "experiment", "experimental", "laboratory", "lab",
    "vivisection", "guinea pig", "test subject", "test subjects", "human experimentation",
    "chemical", "biological", "plague", "disease", "virus", "atomic",
    "nuclear", "uranium", "radiation", "prototype", "project", "test",
    "prisoner", "prisoner of war", "prison", "burial", "buried", "skeleton",
    "bones", "grave", "exhumed", "executed", "massacre", "holocaust",
    "legend", "myth", "mythical", "folk tale", "folklore", "forgotten",
    "bunker", "warhead", "conspiracy", "ancient", "medieval",
    "ambush", "ambushed", "siege", "firefight", "combat", "invasion",
    "invaded", "occupied", "war story", "war stories", "operation", "raid",
    "air strike", "airstrike", "assault", "battalion", "squad", "outnumbered",
    "last stand", "kill zone", "counterattack", "guerrilla", "veteran",
    "trench", "trenches", "no man's land", "poison gas", "gas mask",
    "western front", "eastern front", "blitzkrieg", "d-day", "pacific war",
]

# Military context alone is NOT enough — a story/experiment keyword must also match.
# This list is only used as a body-signal for text subs.
NICHE_CONTEXT = [
    "ww1", "wwi", "ww2", "wwii", "world war", "world war i", "world war ii",
    "military", "army", "navy", "aircraft", "tank", "fighter", "bomber",
    "submarine", "battleship", "weapon", "bomb", "missile", "nazi", "hitler",
    "soldier", "war", "battle", "soviet", "german", "japanese", "british",
    "american", "marine", "infantry", "platoon", "patrol", "convoy",
    "campaign", "conflict", "stalin", "churchill", "axis", "allies",
    "pacific", "europe", "front", "division", "regiment",
]


def _min_selftext(c: dict) -> int:
    if c.get("source") == "wikipedia":
        return 500
    if c.get("source") == "llm":
        return 40  # summaries are short by design
    if c.get("source", "").startswith("channel:"):
        return 500
    sub = c["source"].split(":", 1)[-1]
    if sub in TITLE_BASED_SUBS:
        return 40
    if sub in TRUSTED_SUBS:
        # The title carries the story; the body only has to exist. The
        # research stage re-sources the subject from scratch anyway, so a
        # short post must not gate a strong headline.
        return 80
    return MIN_SELFTEXT


def _min_score(c: dict) -> int:
    if c.get("source") == "wikipedia":
        return 0
    if c.get("source") == "llm":
        return 0
    if c.get("source", "").startswith("channel:"):
        return 0
    sub = c["source"].split(":", 1)[-1]
    floor = REDDIT_SCORE_FLOORS.get(sub)
    if floor is not None:
        return floor
    if sub in TITLE_BASED_SUBS:
        return 200
    return MIN_SCORE


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
    # Hard niche gate: must be military / experiments / dark history, not
    # crime — except on the subs where crime IS the topic.
    sub = c["source"].split(":", 1)[-1]
    if sub not in CRIME_EXEMPT_SUBS:
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

    if c.get("source") == "llm":
        # The LLM is prompted to stay in-niche and grounded; trust the proposal.
        return True

    if c.get("source", "").startswith("channel:"):
        # Channel seeds are already LLM-gated to the niche (mystery/scary/injustice/heroic).
        return True

    sub = c["source"].split(":", 1)[-1]
    if sub in TRUSTED_SUBS:
        # The sub defines the niche; NICHE_STRONG is WW2/legend wording and
        # would reject every frontier, true-crime and ancient title.
        return True
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


def _token_overlap(a: str, b: str) -> float:
    """Fraction of the shorter title's tokens that appear in the other."""
    ta, tb = set(_normalize(a).split()), set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _is_duplicate(title: str, used: list, fuzzy: bool = True) -> bool:
    """Exact or fuzzy match against used topics.

    Fuzzy: if >70% of the shorter title's tokens are shared, treat as a
    duplicate (catches "Tuskegee Syphilis Study" vs "The Tuskegee Study").
    """
    norm = _normalize(title)
    for t in used:
        if _normalize(t) == norm:
            return True
        if fuzzy and len(norm.split()) >= 2 and _token_overlap(title, t) >= 0.6:
            return True
    return False


def _scan_made_videos() -> list:
    """Return topic titles reconstructed from existing output/*.mp4 filenames.

    The output dir is the ground truth of videos actually made — this catches
    duplicates even if used_topics.json was deleted or never written.
    """
    made = []
    if os.path.isdir(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".mp4"):
                slug = f[:-4].replace("_", " ")
                made.append(slug)
    return made


# ---------------------------------------------------------------- output fields

def _verification_status(c: dict) -> str:
    if c.get("source", "").startswith("channel:"):
        # LLM-gated from a real documentary transcript; topics are still on-niche.
        return "verified"
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


def _rank_key(c: dict):
    if c.get("source", "").startswith("channel:"):
        return (0, 4, c["score"])  # by views — proven viral topics surface first
    if c.get("source", "").startswith("reddit:"):
        # Engagement-verified stories are the primary source now, so they rank
        # above the unproven LLM proposals and above the Wikipedia backstop.
        return (0, 3, c["score"])
    if c.get("source") == "llm":
        return (0, 2, c["score"])  # grounded but unproven
    if c.get("source") == "wikipedia":
        return (0, 1, c["title"])  # backstop source
    return (0, 0, c["score"])


def _save_used(titles: list) -> None:
    with open(USED_TOPICS_FILE, "w") as f:
        json.dump(list(dict.fromkeys(titles)), f, indent=2)


def record_made_video(topic_title: str) -> None:
    """Record a successfully made video so it is never regenerated.

    Called after a render completes. Mirrors the used_topics.json record
    (so the exact topic is blocked) and is reconciled with output/ on the
    next generation run regardless.
    """
    with _used_lock:
        used = list(dict.fromkeys(_load_used() + [topic_title]))
        _save_used(used)


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