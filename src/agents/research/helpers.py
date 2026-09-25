"""Deterministic research-pipeline helpers: vetting, source layering, parsing.

No LLM/network here — the model calls live in ``agent.py``. The niche map,
vetting rules, JSON coercers, and the Wikipedia source fetch are shared so the
deep pipeline and the light idea pass behave identically.
"""
import json
import re

from src.agents.topic.wikipedia import (
    _wiki_category_members,
    _wiki_extract_many,
    _wiki_search,
)

TOPICS_OUTPUT_DIR = "topics"

# ---------------------------------------------------------------- vetting

# Hard reject: true crime / serial killers / missing persons — still out of
# scope. Paranormal, cryptid and folklore are NOT rejected: scary stories and
# legends (ghosts, curses, hauntings, unexplained creatures) are part of the
# niche, as long as they read as a legend/story rather than true crime.
# "was killed" is intentionally ABSENT — casualty language is the substance of
# war-story topics.
VET_BLOCK = [
    "serial killer", "murder", "rapist", "stalker", "cctv",
    "kidnap", "abduction", "missing person", "body found",
    "school shooting", "mass shooter", "arrested", "police", "criminal",
    "manslaughter", "human trafficking", "drug cartel", "cartel", "gang",
    "gangster",
]

# Angle classifier: mystery | scary | injustice | heroic
ANGLE_MYSTERY = ["secret", "mystery", "mysterious", "unexplained", "hidden",
                 "cover-up", "cover up", "classified", "vanished", "disappeared",
                 "lost", "unknown", "forgot", "forgotten", "conspiracy",
                 "declassified", "code", "cipher", "coded"]
ANGLE_SCARY = ["horror", "terrify", "horrify", "fear", "dread", "nightmare",
               "death", "dead", "die", "killed", "disaster", "collapse",
               "crash", "explosion", "massacre", "slaughter", "doomed",
               "trapped", "suffocating", "burned", "drowned"]
ANGLE_INJUSTICE = ["injustice", "betrayal", "cover-up", "cover up", "neglect",
                   "cost-cutting", "cost cutting", "greed", "corruption",
                   "abuse", "exploited", "victims", "innocent", "suppressed",
                   "silenced", "forgotten victims", "lies", "lie", "lied",
                   "scandal", "conspiracy against"]
ANGLE_HEROIC = ["hero", "heroic", "bravery", "brave", "stood", "holding",
                "underdog", "last stand", "against", "defended", "defense",
                "saved", "rescued", "legend", "impossible", "outnumbered"]

ANGLE_RULES = [
    ("injustice", ANGLE_INJUSTICE),
    ("scary", ANGLE_SCARY),
    ("heroic", ANGLE_HEROIC),
    ("mystery", ANGLE_MYSTERY),
]


def _vet_idea(idea: dict) -> dict:
    """Reject off-niche ideas and tag an angle. Returns None if rejected."""
    text = f"{idea['title']} {idea.get('hook', '')} {idea.get('why_viral', '')}".lower()
    if any(b in text for b in VET_BLOCK):
        return None
    # Modern-era items (20xx) are off-niche: we produce WW1/WW2 stories,
    # history-rooted legends, and 20th-century experiments, not current-events
    # or pop-culture internet lore.
    if re.search(r"\b(?:20[0-9]{2}|19[89][0-9])\b", text):
        return None
    for angle, keywords in ANGLE_RULES:
        if any(k in text for k in keywords):
            idea["angle"] = angle
            return idea
    # No signal — default to mystery (the channel's core pillar).
    idea["angle"] = "mystery"
    return idea


# ---------------------------------------------------------------- niches

# The channel's pillars, in priority order:
#   1. Classified human/animal experiments run by the great powers in the
#      world wars (Unit 731, Nazi medicine, chemical weapons test subjects…).
#   2. Dark legends, scary stories, cover-ups, curses and hauntings.
#   3. Epic true stories of WW1 / WW2 — the battles and the people in them.
NICHES = {
    # The money niche: crazy, brutal, off-the-books experiments on humans and
    # animals run by big powers around WW1/WW2.
    "experiments": {
        "categories": [
            "Category:Japanese human subject research",
            "Category:Nazi human subject research",
            "Category:Medical experimentation on prisoners",
            "Category:Human subject research",
            "Category:Medical ethics",
            "Category:Biological warfare",
        ],
        "queries": [
            "Unit 731 biological warfare experiments",
            "Nazi medical experiments human subjects",
            "WW2 human experimentation prisoners",
            "Japanese war crimes vivisection",
            "chemical weapon testing on soldiers",
            "radiation experiments humans 1940s",
            "animal experiments military secret",
        ],
        "prompt": (
            "human and animal experiments run by the great powers in WW1/WW2: "
            "Unit 731's biological weapons, Nazi medical atrocities, chemical "
            "and radiation tests on prisoners and soldiers, and the secret "
            "labs that treated living people as specimens"
        ),
    },
    # Scary stories, legends, cover-ups and hauntings tied to real history.
    "dark_legends": {
        "categories": [
            "Category:Urban legends",
            "Category:Folklore",
            "Category:Curses",
            "Category:Paranormal",
            "Category:Conspiracy theories",
        ],
        "queries": [
            "famous urban legend true story",
            "haunted place curse history",
            "unsolved mystery cover-up",
            "legend disappeared soldier unit",
            "cursed object curse history",
            "secret men in black story",
            "vanished submarine mystery",
        ],
        "prompt": (
            "scary stories, legends, curses, hauntings and cover-ups that are "
            "rooted in real history — the unexplained, the hidden, and the "
            "things the authorities wanted forgotten"
        ),
    },
    # Epic, dramatic true stories of the world wars: battles, last stands,
    # impossible rescues, and the soldiers caught in them.
    "ww1_ww2_stories": {
        "categories": [
            "Category:Battles of World War I",
            "Category:Campaigns of World War I",
            "Category:Battles and operations of World War II",
            "Category:Campaigns of World War II",
            "Category:People of World War II",
            "Category:Last stands",
        ],
        "queries": [
            "World War 1 last stand battle",
            "World War 2 outnumbered battle",
            "WW1 trench raid story",
            "WW2 impossible rescue mission",
            "Verdun Somme battle story",
            "Stalingrad battle story",
            "Battle of Britain pilot story",
        ],
        "prompt": (
            "epic true stories from World War I and World War II: desperate "
            "last stands, outnumbered units fighting impossible odds, "
            "daring rescues and escapes, and the ordinary men and women "
            "thrown into the biggest wars in history"
        ),
    },
}


# ---------------------------------------------------------------- source layer

def _fetch_niche_sources(niche: str, limit: int = 25) -> list:
    """Gather raw candidate articles for one niche (title + url + intro)."""
    spec = NICHES[niche]
    titles = []
    seen = set()

    for cat in spec["categories"]:
        try:
            for t in _wiki_category_members(cat, limit=30):
                if t.lower() not in seen:
                    seen.add(t.lower())
                    titles.append(t)
        except Exception:
            continue

    for q in spec["queries"]:
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
        sources.append({
            "title": t,
            "url": f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}",
            "content": ext,
        })

    if not sources:
        print(f"  [source] {niche}: nothing usable")
    return sources


# ---------------------------------------------------------------- JSON coercion

# Max sources per single LLM call, and max chars per source, so an individual
# request stays well under the 8k-TPM free-tier ceiling of Groq/Ollama.
# (Chunked, not RAG-indexed: the goal is small prompt windows, not retrieval.)
IDEAS_CHUNK_SOURCES = 4
IDEAS_SOURCE_CHARS = 500


def _parse_idea_json(raw_out: str) -> list:
    """Strip fences/prose, grab the JSON array, normalize to idea dicts.

    Prefers the outermost top-level JSON array; falls back to line-by-line JSONL
    so a TRUNCATED array / interrupted response still yields completed ideas.
    """
    if not raw_out:
        return []
    raw = raw_out.strip()
    if raw.startswith("```"):  # strip markdown fences
        raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw).strip()

    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [x for x in (_normalize_idea_item(i) for i in data) if x]
        except Exception:
            pass
    # JSONL first (line-aligned objects) — before the outermost-array scan so an
    # inner helper array (e.g. "outline": [...]) never shadows real idea objects.
    ideas = _jsonl_fallback(raw)
    if ideas:
        return ideas
    # Outermost balanced array (model wrapped the array in prose).
    outer = _outermost_array(raw)
    if outer:
        try:
            data = json.loads(outer)
            if isinstance(data, list):
                return [x for x in (_normalize_idea_item(i) for i in data) if x]
        except Exception:
            pass
    return []


def _outermost_array(text: str):
    """Return the OUTERMOST [...] block as a string (ignores inner arrays)."""
    start = text.find("[")
    if start < 0:
        return None
    depth, i = 0, start
    while i < len(text):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    return None


def _jsonl_fallback(raw_out: str) -> list:
    """Parse one JSON object per line when no complete array exists."""
    ideas = []
    for line in raw_out.splitlines():
        line = line.strip()
        if not (line.startswith("{") or line.startswith("[")):
            continue
        for cand in (line, line.rstrip(",")):
            try:
                item = json.loads(cand)
                break
            except Exception:
                item = None
        if isinstance(item, list) and item:
            item = item[0]
        idea = _normalize_idea_item(item)
        if idea:
            ideas.append(idea)
    return ideas


def _normalize_idea_item(item) -> dict:
    """Coerce one raw JSON dict into a valid idea dict (None if unusable)."""
    if not isinstance(item, dict):
        return None
    title = str(item.get("title", "")).strip()
    if not title:
        return None
    try:
        confidence = min(10, max(1, int(item.get("confidence", 5))))
    except (TypeError, ValueError):
        confidence = 5
    return {
        "title": title,
        "hook": str(item.get("hook", "")).strip(),
        "why_viral": str(item.get("why_viral", "")).strip(),
        "outline": item.get("outline", []),
        "source_url": str(item.get("source_url", "")).strip(),
        "confidence": confidence,
    }


# ---------------------------------------------------------------- light mode

# First-principles topic gen: a script is just an interesting combination of
# sentences, and interestingness comes from how it is WRITTEN (hooks, the
# writing-style techniques), not from ranking sources to find a "viral idea".
# So the light path does zero source scraping, zero scoring, zero ranking,
# zero channel mining: one LLM call per niche proposes titles, we take the
# FIRST `target` titles we have not already made a video about, and produce.

LIGHT_SUBJECTS = {
    "experiments": "secret WW1/WW2 human and animal experiments by the great powers",
    "dark_legends": "dark legends, curses, hauntings and scary true history",
    "ww1_ww2_stories": "epic stories from World War I and World War II",
}


def _parse_titles(raw_out: str) -> list:
    """Parse a JSON string-array reply (with fences/prose stripped)."""
    if not raw_out:
        return []
    raw = raw_out.strip()
    raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x) and str(x).strip()]
    except Exception:
        pass
    outer = _outermost_array(raw)
    if outer:
        try:
            data = json.loads(outer)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x) and str(x).strip()]
        except Exception:
            pass
    return []


# ---------------------------------------------------------------- scoring

def _score_idea(idea: dict, angle_map: dict) -> dict:
    """Novelty (confidence) is the controllable proxy for virality."""
    score = idea["confidence"] * 10
    # Slight lift for ideas backed by a real source URL (verifiable).
    if idea.get("source_url"):
        score += 5
    return score