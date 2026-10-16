"""Topic-level search-entity extraction for image sourcing.

WHY THIS EXISTS
---------------
Narration lines are deliberately entity-free: short-form retention writing
strips proper nouns ("Hours later, the sky tore open with the roar of
low-flying aircraft."). Measured on 33 real pipeline lines, 30 (90.9%) carry
no searchable entity. So per-line entity extraction is a dead end.

The proper nouns live in the TOPIC (title + article body), not the line. This
module builds that topic-level entity pool once per video; the per-line
relevance scoring elsewhere decides which pool image lands on which line.

Search engines that index archival photo FILENAMES (Wikimedia Commons) return
10/10 hits for "USS Enterprise CV-6" and 0/10 for the full prose headline.
So the pool is built from short, entity-shaped strings, never prose.

The LLM call is one per video (default gemma3:1b via the existing local
Ollama seam). A deterministic regex backstop guarantees a usable pool when
the model is unavailable, times out, or returns junk.
"""
from __future__ import annotations

import json
import re

from src.agents.common.llm import _local

# Generic narration words that must never become a search term.
_GENERIC = set("""
war wars battle battles fought fight fighting army armies navy military soldier
soldiers troop troops plane planes aircraft ship ships boat vessel carrier carriers
bomb bombs bombing attack attacks attacked weapon weapons fire fires burning kill
killed kills death deaths died dead body bodies wound battlefront front line troops
man men woman women child children family people person crowd group young old
country nation land water sea ocean sky night day morning afternoon evening time
year years today yesterday tomorrow history story histories secret secrets mystery
mysterious hidden lost forgotten unknown unexplained discovered found new first
last one two three four five several many few more most other another same
""".split())

# Prose/editorial scaffolding in our own headlines ("How the US Navy Found...").
_EDITORIAL = set("""
how why what when where who which whose that this these those explained
explaining explain telling told story stories legacy hidden secrets inside
forgotten untold true real actual mystery mysterious scandal cover-up coverup
""".split())

_ENTITY_PROMPT = """Extract proper nouns from this history article that would help \
find relevant historical photographs in the Wikimedia Commons archive.

Rules:
- ONLY specific named things: ships, aircraft, units, weapons, places, people, operations.
- For ships, ALWAYS include the full name and hull designation if the text has it, e.g. \
"USS Enterprise (CV-6)".
- Include years.
- Do NOT include generic words (war, battle, army, aircraft, bomb, history).
- Do NOT invent anything that is not in the text.
- Order by how likely the term is to have real archival photographs.

TITLE: {title}
TEXT: {body}

Return ONLY a JSON array of 8-14 strings."""

_YEAR = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")
_HULL = re.compile(
    r"\b(?:USS|RMS|HMS|KMS|IJN|JMS|SMS|CV|BB|DD|CA|CL|SSN)\s*\(?[A-Z]*-?\d+\)?[A-Z]?\b"
)
_CAP_PAIR = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b")
_CAP_WORD = re.compile(r"\b[A-Z][A-Z]{2,}\b")
_QUOTED = re.compile(r"[\"“]([^\"”]{3,40})[\"”]")

# Single Capitalized words are only trustworthy when they are NOT opening a
# sentence: "The Japanese Combined Fleet" starts a sentence, "the Akagi" does
# not. A lone "Akagi" is the entity; a lone "The" is not.
_CAP_SINGLE = re.compile(r"(?<![.!?]\s)(?<!\n\s)\b([A-Z][a-z]{3,})\b")
_SENTENCE_OPENERS = set("""
the a an and or but if when while after before during since until although
because so then thus this that these those there here his her their our your
it he she they we you i as not no in on at to for from by with without within
but also however meanwhile instead yet still now then both either neither
""".split())
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.I)

# Bare ship-class and unit prefixes. "USS" alone matched 5 unrelated naval
# photos; a prefix is only meaningful attached to its name.
_BARE_PREFIX = set("""
uss rms hms kms ijn jms sms uss hmas
cv bb dd ca cl cmc ddg ssn ssbn ssn
british german japanese american italian french russiansoviet
""".split())

# A word that is a fragment of a longer phrase in the pool ("Kido" in
# "Kido Butai") must not also be searched on its own.
_FRAGMENT_PARTS = re.compile(r"^(?:and|of|the|a|an)\s+", re.I)


def _is_fragment(term: str, pool: list) -> bool:
    """True when `term` is contained in, or contains, a longer pool entry."""
    t = term.strip().lower()
    if t in _BARE_PREFIX:
        return True
    for other in pool:
        o = other.strip().lower()
        if o != t and (o in t or t in o):
            return True
    return False


_WEAK_ALONE = set("""
nazi german japanese america american british british soviet russian china
chinese france french germany japan russia china italy roman roman empire
force forces fleet fleets navy armies army air sea today now then
capture captured battle battles war wars history museum archive photo photograph
crew aircrew baron lord lady king queen emperor president general admiral
""".split())

# A term needs at least one of these to be worth searching on its own; Commons
# matches ANY word, so "Force" retrieves polymer-physics diagrams. Measured:
# "Werwolf" -> Werwolf-pool.JPG, "Nazi" -> an antique hat, "Baron" -> a mairie.
_STRONG = set("""
carrier carriers submarine submarines u-boat uboat battleship cruiser destroyer
frigate destroyer corvette schooner trawler tanker freighter
aircraft fighter bomber reconnaissance seaplane zeppelin warship
tank tanks gun guns cannon rifle rifleman torpedo torpedoes mine mines
division squadron regiment brigade corps battalion company infantry cavalry
artillery navy marines airforce Luftwaffe Wehrmacht Kriegsmarine
missile rocket rockets bombs bomber
island islands mountain mountains river valley desert beach port harbour harbor
city cities town village prison camp laboratory hospital university college
camp concentration massacre tribunal trial trials execution executions
mississippi alaska normandy pearl harbor guadalagman atol leningrad stalingrad
nuremberg auschwitz dachau hiroshima nagasaki berlin vienna versailles
""".split())


def _clean(term: str) -> str:
    """Normalize one candidate into a short entity-shaped search string."""
    t = (term or "").strip().strip(".,;:!?'\"“”")
    t = _LEADING_ARTICLE.sub("", t)
    t = re.sub(r"\s+", " ", t)
    if len(t) < 3 or len(t) > 60:
        return ""
    if t.islower() and " " not in t:
        return ""                      # a bare lowercase word is never an entity
    words = [w for w in t.split() if w.lower().strip(",") not in _EDITORIAL]
    if not words:
        return ""
    t = " ".join(words).strip()
    if all(w.lower() in _GENERIC for w in t.split()):
        return ""
    return t


def _worth_searching(term: str) -> bool:
    """Drop terms Commons cannot answer usefully.

    A bare year is never a search term here: "1942" returns harvest photos from
    Devon. A bare nationality is worse: "Nazi" returns a hat. Both must be
    paired with a strong subject word ("Peenemunde 1943", not "1943").
    """
    t = (term or "").strip()
    if not t:
        return False
    if _YEAR.fullmatch(t):
        return False                  # bare year
    words = [w.lower().strip(",") for w in t.split()]
    if all(w in _WEAK_ALONE for w in words):
        return False                  # only generic/nationality words
    if len(words) == 1 and words[0] in _GENERIC:
        return False
    # Multi-word prose built from editorial+generic words is not an entity.
    if len(words) > 2 and all(
        w in _GENERIC or w in _WEAK_ALONE or w in _EDITORIAL for w in words
    ):
        return False
    return True


def _regex_entities(title: str, body: str) -> list:
    """Deterministic backstop: hull numbers, years, proper-noun runs, quotes.

    Multi-word proper-noun runs are emitted ONLY as the whole run. Emitting the
    fragments too flooded the pool: "Butai" alone returned 6 images and
    "Rocket" alone another 6, while the real terms were crowded out. Commons
    matches any word, so a fragment is a valid-but-useless query, not a safe
    fallback.
    """
    text = f"{title}. {body}"
    found = []
    for pat in (_HULL, _YEAR, _CAP_PAIR, _QUOTED, _CAP_WORD):
        found.extend(m.group(0) if m.lastindex is None else m.group(1)
                     for m in pat.finditer(text))
    singles = []
    for m in _CAP_SINGLE.finditer(text):
        w = m.group(1)
        if w.lower() not in _SENTENCE_OPENERS and w.lower() not in _GENERIC:
            singles.append(w)
    out, covered = [], []
    # Whole runs first, then only singles that no run already contains, so
    # "Kido Butai" wins and the orphan "Kido" / "Butai" never become queries.
    for t in found + singles:
        c = _clean(t)
        if not c or not _worth_searching(c):
            continue
        if any(c.lower() in prev.lower() or prev.lower() in c.lower() for prev in covered):
            continue
        covered.append(c)
        if c not in out:
            out.append(c)
    return out


def _parse_array(raw: str) -> list:
    if not raw or not raw.strip():
        return []
    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    for candidate in (txt, *(m.group(0) for m in re.finditer(r"\[.*\]", txt, re.S))):
        try:
            arr = json.loads(candidate)
        except Exception:
            continue
        if isinstance(arr, dict):
            arr = [v for vs in arr.values()
                   for v in (vs if isinstance(vs, list) else [vs])]
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    return []


def _grounded_in(term: str, title: str, body: str) -> bool:
    """Reject entities the model invented.

    gemma3:1b hallucinated "USS Enterprise (CV-6)" into a German U-BOAT
    article and it became that video's single largest image source (6 photos
    of a US carrier under a video about a U-boat). Its output looked perfect,
    so nothing downstream would have caught it. Requiring the term to appear
    literally in the title or body is a deterministic fix.

    Punctuation/parenthesis differences are tolerated: "Enterprise (CV-6)"
    still matches "Enterprise CV-6" in the source.
    """
    hay = f"{title} {body}".lower()
    needle = re.sub(r"[^\w\s]+", " ", term.lower())
    needle = re.sub(r"\s+", " ", needle).strip()
    if not needle:
        return False
    if needle in hay:
        return True
    toks = needle.split()
    # A hull designation may be split or merged between term and source
    # ("USS Enterprise (CV-6)" vs "Enterprise CV-6"), so require every
    # alphanumeric token to be present somewhere in the source.
    return all(t in hay for t in toks if len(t) > 2)


def clean_entities(terms: list, title: str, body: str = "",
                   max_entities: int = 14) -> list:
    """Validate externally-supplied terms with the same filters the model gets.

    The shot planner proposes the topic's real subjects; this guarantees they
    still pass every deterministic gate (not generic/editorial, not a bare year
    or nationality, grounded in the source text, not a fragment of a longer
    term) before they become Commons searches.
    """
    cleaned = []
    for t in terms or []:
        c = _clean(t)
        if c and _worth_searching(c) and _grounded_in(c, title, body):
            cleaned.append(c)
    out, seen = [], set()
    for c in cleaned:
        k = c.lower()
        if k in seen or _is_fragment(c, out):
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= max_entities:
            break
    return out


def extract_entities(title: str, body: str = "", max_entities: int = 14) -> list:
    """Topic -> ordered list of short, entity-shaped Commons search terms.

    One local LLM call; falls back to regex-only when the model is missing,
    times out, or returns nothing usable. Every candidate must be (a) not a
    fragment of another term, (b) not a bare prefix/generic, and (c) literally
    present in the source text. Never raises: an empty pool is handled
    downstream by the existing keyword/backfill paths.
    """
    body = (body or "")[:3000]
    pool = []
    try:
        raw = _local(_ENTITY_PROMPT.format(title=title, body=body),
                     temperature=0.1, tag="entities",
                     model="gemma3:1b", repeat_penalty=1.0, local_only=True)
        pool = _parse_array(raw)
    except Exception as e:  # noqa: BLE001
        print(f"  [entities] local model unavailable ({type(e).__name__}) - regex only")

    # Clean + filter first, then drop fragments relative to the surviving pool.
    return clean_entities(pool + _regex_entities(title, body), title, body,
                          max_entities)
