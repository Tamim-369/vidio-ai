"""Per-image validation. Runs in parallel, before assignment.

Why this exists: a rendered video came out as "a photo of an island and a bunch
of white birds", and later as "one random building image with 'pecifc' written
on it". Both were term-level faults that no query filter could see:

  * "Midway" means the atoll, so every image was a beach and some seabirds.
  * "Pacific Theater" is a region, not an object, so Commons returned an
    operations-theatre map, and a command building.
  * "USS Lexington (CV-3)" is the WRONG ship -- CV-3 was Saratoga, Lexington was
    CV-2 -- and its images are the Corpus Christi museum on the bay.

Grounding already guarantees the term occurs in the article. It cannot tell a
place from an object, a museum from the thing in it, or a map from a photo. This
module answers those questions from Commons' own metadata, which the fetcher
already had and discarded.

Design notes:
  * Every check is a score, not a hard rule, because no single signal is
    reliable. Categories flagged a genuine 1942 action shot as "Dioramas" while
    passing a 2004 photo of a fleet command building, so categories alone
    cannot decide.
  * REJECT only on a confident combination; otherwise keep and let assignment
    rank. A false reject empties the pool for a narrow topic; a false accept
    costs one image.
  * Runs on metadata only. No pixels are downloaded to validate, which keeps it
    free and fast.
"""
from __future__ import annotations

import re

# --- what Commons calls non-photographic subject matter --------------------
# Maps, diagrams and charts are never the subject of a history short.
# NOT "graphs": Commons categories routinely end in "... photographs", and
# "graphs" matched that substring, so every genuine archival photograph was
# rejected as a map and real pools came back empty. Match whole words only.
_MAPY = re.compile(
    r"\bMaps?\b|\bMap of\b|Maps of|Diagrams?\b|Charts?\b|\bGraphs\b|"
    r"Line drawings|Timetables?\b|Infographics?\b|Schematics?\b|"
    r"\bPlans? of\b|Blueprints?\b|Cartography\b|Gazetteers?\b|Atlases\b|"
    r"Historical maps|Locator maps",
    re.I,
)

# A museum, a memorial, a monument: a building ABOUT the subject. Usable as a
# establishing shot, but if it is the only shot the video is a building.
_PLACES = re.compile(
    r"Museum ships?|Views? from [^|]*Museum|Museums? in |"
    r"Exteriors? of (?:museum|church|temple|castle)|"
    r"Interiors? of (?:museum|church|temple)|Suburbs?\b|",
    re.I,
)

# Institutional graphics, not photographs of events.
_EMBLEMS = re.compile(
    r"Logos?\b|emblems?\b|Coats? of arms|Seals? of|Crests?\b|"
    r"Banners?\b|Flags? of|Insignia\b|Patches\b",
    re.I,
)

# Textual artefacts: a scan of a book, a plaque, a gravestone, a chart caption.
_TEXTUAL = re.compile(
    r"Graffiti\b|Street art|Inscriptions?\b|Plaques?\b|Gravestones?\b|"
    r"Epitaphs?\b|Tombstones?\b|Painted walls?|Murals?\b|"
    r"Newspapers?\b|Headlines?\b|Postcards?\b|Stamps?\b|"
    r"Documents?\b|Manuscripts?\b|Letters?\b|Typewritten|"
    r"Text(?:s)? of|Captions?\b|Cover art|Book pages|Page(?:s)? from",
    re.I,
)

# Replica / model / prop: a representation rather than the thing.
_REPLICA = re.compile(
    r"\bDioramas?\b|Models?\b|Minatures?\b|Replica|Replicas|"
    r"Scale models|Mock-?ups?|Animatronics\b",
    re.I,
)

# A "modern" subject photograph is a tourist photo of a preserved ship, not a
# wartime one. Kept separate from _PLACES so a preserved-warship photo can be
# allowed if nothing else objects.
_MODERN_TOURIST = re.compile(
    r"Tourists?\b|Visitors?\b|Sightseeing\b|Watermark|"
    r"Modern (?:museum|tourist)|\b20[0-2]\d photographs\b",
    re.I,
)

_PLACE_TITLE = re.compile(
    r"\bMuseum\b|\bMemorial\b|\bMonument\b|\bStatue\b|\bSculpture\b|"
    r"\bCathedral\b|\bChapel\b|\bTemple\b|\bPalace\b|\bCastle\b|"
    r"\bBuilding\b|\bFacade\b|\bPlaque\b|\bGraffiti\b",
    re.I,
)

_BAD_CATS = (_MAPY, _PLACES, _EMBLEMS, _TEXTUAL, _REPLICA)

# An entity that names a place, region, command or abstract group rather than a
# physical object. These produce maps and buildings even when the term is a
# perfect match for the article.
NON_VISUAL_ENTITIES = {
    "pacific theater", "pacific fleet", "uss pacific fleet",
    "southwest pacific", "western pacific", "eastern pacific",
    "united states pacific fleet", "navy", "us navy", "united states navy",
    "royal navy", "imperial japanese navy", "japanese navy",
    "combined fleet", "japanese combined fleet", "task force",
    "theater of operations", "european theater", "atlantic theater",
    "army", "us army", "united states army", "air force", "marine corps",
    "pacific war", "world war", "world war ii", "world war 2",
    "the war", "the battle", "naval history", "military history",
    "united states", "america", "japan", "germany", "britain", "france",
    "north pacific", "pacific ocean", "atlantic", "mediterranean",
}

# Words that mark a term as a place/command even in a longer phrase.
_NON_VISUAL_WORDS = re.compile(
    r"\b(?:theater|theatre)\b|\bfleet\b|\barmy\b|\bnavy\b|\bcommand\b|"
    r"\bregiment\b|\bdivision\b|\bsquadron\b|\bbrigade\b|\bcorps\b|"
    r"\boperation\b|\bcampaign\b|\bwar\b|\bbattle\b|\bconflict\b|"
    r"\bmission\b|\bzone\b|\barea of operations\b|\btheater of operations\b",
    re.I,
)

_YEAR = re.compile(r"\b(1[5-9]\d\d|20[0-4]\d)\b")


def is_non_visual_entity(entity: str) -> bool:
    """True when a term names a place, command or abstract group, not an object."""
    e = (entity or "").strip().lower()
    if not e:
        return True
    base = re.sub(r"^(?:uss|mss|ijms|rms|hms)\s+", "", e)
    base = re.sub(r"\s*\([^)]*\)\s*$", "", base).strip()
    if e in NON_VISUAL_ENTITIES or base in NON_VISUAL_ENTITIES:
        return True
    # "the Pacific Theater of World War II", "5th Fleet", "US 3rd Army"
    if _NON_VISUAL_WORDS.search(e) and not re.search(r"\b(ship|carrier|submarine|"
                                                     r"destroyer|cruiser|battleship|"
                                                     r"aircraft|plane|squadron of|"
                                                     r"portrait|painting)\b", e, re.I):
        return True
    # A bare date is not a picture of anything: "December 7 1941" searches for
    # commemorative events, not for the attack.
    if re.search(r"\b(january|february|march|april|may|june|july|august|"
                 r"september|october|november|december)\b", e):
        return True
    if _YEAR.fullmatch(e):
        return True
    # A bare hull number with no name ("CV-3") has no subject to match on, and
    # Commons answers it with unrelated carriers and museum visits.
    if re.fullmatch(r"(?:[a-z]{1,4}\s*-?\s*)?\d+", e):
        return True
    if re.fullmatch(r"[\d\s,.\-–—:]+", e):
        return True
    return False


def _cat_hits(pattern, cats: str) -> str:
    """First segment matching `pattern`, or "". Category lists are
    pipe-delimited, so each segment is matched on its own.
    Always returns "" rather than a zero-width match, which would
    score a penalty for a pattern that did not really hit."""
    for seg in (cats or "").split("|"):
        m = pattern.search(seg)
        if m and m.group(0).strip():
            return m.group(0).strip()
    return ""


_HULL = re.compile(r"\b([A-Z]{2,4})-?(\d{1,3})\b")


def hull_designator(entity: str) -> tuple[str, str] | None:
    """Split a term into (ship name, hull number).

    "USS Lexington (CV-3)" -> ("uss lexington", "cv-3"). The name is stripped of
    any trailing "(" so it can be matched against a Commons category.
    """
    m = _HULL.search(entity or "")
    if not m:
        return None
    name = (entity or "")[: m.start()].strip().rstrip("([").strip().lower()
    hull = m.group(0).replace(" ", "").lower()
    return (name, hull) if name else None


def resolve_hull(entity: str, cats_of_results: list[str]) -> str:
    """Correct a term whose hull number disagrees with Commons.

    Commons' own category names carry the true designation, so a hallucinated
    one can be detected without a ship database. "USS Lexington (CV-3)" is
    wrong -- CV-3 was Saratoga -- and every result comes back categorised
    "USS Lexington (CV-2)" or "(CV-16)". Trusting the query would have shipped
    the wrong ship.

    Only rewrites when the named ship is unambiguous: we need at least two
    results agreeing on a different hull, and the queried hull must be absent
    from all of them.
    """
    parsed = hull_designator(entity)
    if not parsed:
        return entity
    name, queried = parsed
    tally: dict[str, int] = {}
    # Commons writes the category as "USS Lexington (CV-2)" -- no space before
    # the parenthesis -- so both spellings must be accepted.
    pat = re.escape(name) + r"\s*\(\s*([a-z]{2,4}-\d{1,3})\s*\)"
    for cats in cats_of_results:
        for m in re.finditer(pat, (cats or "").lower()):
            tally[m.group(1)] = tally.get(m.group(1), 0) + 1
    if queried in tally or not tally:
        return entity                      # query agrees, or Commons says nothing
    # Two different hull numbers for one name is not ambiguity -- it means the
    # queried hull is simply unattested. Trust the one most of Commons agrees
    # with, needing at least two results to move off the original term.
    best = max(tally.items(), key=lambda kv: kv[1])
    if best[1] < 2:
        return entity
    # Reuse the original casing: .title() would render "Uss Lexington".
    head = (entity or "")[: _HULL.search(entity).start()].strip().rstrip("( ")
    return f"{head} ({best[0].upper()})"


def _meta(meta: dict, key: str) -> str:
    v = ((meta or {}).get("extmetadata") or {}).get(key) or {}
    val = v.get("value", "")
    if not isinstance(val, str):
        val = str(val)
    return re.sub(r"<[^>]+>", " ", val)


def validate_candidate(meta: dict, title: str, entity: str,
                      era_year: int | None = None) -> dict:
    """Score one candidate from Commons metadata alone.

    meta: the imageinfo dict (needs extmetadata); title: the file title;
    entity: the search term it came from; era_year: the topic's year, if known.

    Returns {"verdict": "keep"|"reject", "score": int, "reasons": [str]}.
    A negative score accumulates evidence against the image; anything at or
    above 0 is kept, so this only rejects when several signals agree.
    """
    cats = _meta(meta, "Categories")
    score, reasons = 0, []
    type_score = 0

    hit = _cat_hits(_MAPY, cats)
    if hit:
        score -= 4
        type_score -= 4
        reasons.append(f"map/chart: {hit!r}")

    # A place is judged by its own categories AND by the file title, because
    # the categories of a genuine archival photo name the archive holding it
    # ("DPLA", "NARA", "Naval History and Heritage Command") while its title
    # still names the ship.
    place_hit = _cat_hits(_PLACES, cats)
    title_hit = _PLACE_TITLE.search(title or "")
    if place_hit or title_hit:
        score -= 3
        type_score -= 3
        reasons.append(f"place not object: {(place_hit or title_hit.group(0))!r}")
    if _EMBLEMS.search(cats):
        # A fleet emblem is not a photograph of a battle. Strong on its own:
        # "Commander US Pacific Fleet logo" matched no other signal, so an
        # emblem at -2 fell under the reject threshold and shipped a logo.
        score -= 4
        type_score -= 4
        reasons.append("emblem/logo")
    hit = _cat_hits(_TEXTUAL, cats)
    if hit:
        score -= 3
        type_score -= 3
        reasons.append(f"textual artefact: {hit!r}")
    if _REPLICA.search(cats):
        score -= 2
        reasons.append("replica/model/diorama")
    if _MODERN_TOURIST.search(cats):
        score -= 2
        reasons.append("modern tourist photo")

    # Era. DateTimeOriginal is the artwork date and is the best signal we have;
    # DateTime is the upload date and is ignored for era on purpose.
    dt = _meta(meta, "DateTimeOriginal")
    years = _YEAR.findall(dt)
    if years and era_year:
        try:
            made = int(years[0])
        except ValueError:
            made = None
        if made and made > era_year + 40:
            score -= 3
            reasons.append(f"made {made}, topic ~{era_year}")

    # A subject named in the file title is strong positive evidence, because it
    # means Commons itself matched the entity rather than a related concept.
    t_low = (title or "").lower()
    ent_tokens = [t for t in re.split(r"[^\w]+", (entity or "").lower()) if len(t) > 3]
    if ent_tokens and sum(1 for t in ent_tokens if t in t_low) >= max(1, len(ent_tokens) // 2):
        score += 2

    # Very low resolution or extreme aspect is unusable as a video visual.
    try:
        w, h = (meta or {}).get("width", 0), (meta or {}).get("height", 0)
        if w and h and (w < 500 or h < 400):
            # Unusable as a 1080x1920 video visual, and reuploads of small
            # originals are common for museums and memorials.
            score -= 4
            reasons.append(f"too small: {w}x{h}")
    except (TypeError, ValueError):
        pass

    # A type mismatch is categorical and cannot be bought off by a name match.
    # "Pearl Harbor Memorial" scores a title-match bonus, and letting that
    # cancel the type penalty shipped a memorial in a video about the attack.
    # Type problems cap the score; only the name bonus is discountable.
    if type_score <= -3:
        score = min(score, type_score)
    verdict = "reject" if score <= -3 else "keep"
    return {"verdict": verdict, "score": score, "reasons": reasons}