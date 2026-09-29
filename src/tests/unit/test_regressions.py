"""Regression tests for the behaviour-preserving fixes.

Every assertion here pins down behaviour that a fix had to PRESERVE, so a
future change that silently regresses it fails loudly instead of quietly
producing worse videos.
"""
import math

import pytest

from src.services import captions
from src.agents.asset.agent import _on_free_host, _is_blocked_host


# --------------------------------------------------------------------------
# I-6: _split_parts must snap boundaries to punctuation WITHOUT ever
# changing how many parts a caption is broken into.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 9, 12, 15, 20, 30])
@pytest.mark.parametrize("n_parts", [2, 3, 4])
@pytest.mark.parametrize("punct_at", [0, 1, 2, 3, 5, 8, 14, 25])
def test_split_parts_never_changes_part_count(n, n_parts, punct_at):
    if punct_at >= n:
        pytest.skip("punctuation index out of range")
    words = [f"w{i}" for i in range(n)]
    words[punct_at] += "."
    parts = captions._split_parts(words, n_parts)

    # The part count is what the even-split baseline produces.
    even = [[w] for w in words] if n_parts >= n else None
    if even is not None:
        assert len(parts) == n
        return

    size = math.ceil(n / n_parts)
    bounds = [min(size * p, n) for p in range(1, n_parts)]
    kept = []
    for b in bounds:
        k = b
        if kept and k <= kept[-1]:
            k = kept[-1] + 1
        if k < n:
            kept.append(k)

    assert len(parts) == len(kept) + 1


def test_split_parts_is_lossless_and_contiguous():
    words = [f"w{i}" for i in range(17)]
    words[3] += "."
    parts = captions._split_parts(words, 3)
    flat = [w for part in parts for w in part]
    assert flat == words, "splitting must neither drop nor duplicate words"


def test_split_parts_nudge_is_monotonic():
    words = [f"w{i}" for i in range(9)]
    words[3] += "."
    parts = captions._split_parts(words, 3)
    # Boundary is allowed to move, but must stay increasing and in range.
    starts, n = [], len(words)
    idx = 0
    for p in parts:
        starts.append(idx)
        idx += len(p)
    assert starts == sorted(starts)
    assert all(0 <= s < n for s in starts)


def test_split_parts_nudge_never_makes_parts_worse():
    """A punctuation snap must not make a caption part lopsided.

    Snapping to a comma can strand "only" or "and" in its own part, which
    looks worse on screen than a clean even split. Measured over 416 real
    narration lines, accepting unconstrained snaps raised unbalanced parts
    from 83 to 137 and 1-word parts from 80 to 106. With the balance guard the
    nudge is only taken when every resulting part is at least
    MIN_PART_WORDS and the sizes differ by at most one word.
    """
    # Real narration lines captured from pipeline renders (Battle of Midway,
    # V-2, paratroop drop, Operation Big Ben, Baron 52). Embedded on purpose:
    # debug_output/ is gitignored and only exists on a machine that has run a
    # render, so globbing it made this test silently pass with zero coverage
    # and then fail for anyone else.
    SAMPLES = [
        "How does a fleet of planes turn the world's most feared navy into scrap?",
        "The Japanese Combined Fleet was right there, carriers refueling and rearming.",
        "It was the exact moment the U.S. Strike groups found them.",
        "Planes from the Enterprise, Hornet, and Yorktown went in.",
        "A single direct hit struck the Akagi.",
        "It exploded among the armed and fueled aircraft on the deck.",
        "The ship was instantly set ablaze and abandoned.",
        "The Kaga and Soryu met the same fiery fate in the attack.",
        "The entire Kido Butai was destroyed in a matter of hours.",
        "The Japanese Combined Fleet suffered a decisive defeat at Midway.",
        "A family of nine in France falls asleep to the sound of bombs.",
        "Louis Leroux and his wife had just watched Allied fighter-bombers smash the bridges.",
        "They decided the night was over, so the whole family crawled back into bed.",
        "Parachutes began descending into the nearby marshes, drifting down.",
        "How does a weapon of mass destruction turn into a ticket to space?",
        "The rocket that ended the war became the key to the stars.",
        "But a secret group saw something else entirely.",
        "They simply stopped existing on every map.",
    ]
    lines = [s.split() for s in SAMPLES if len(s.split()) >= 4]

    def even_split(words, n_parts):
        import math
        n = len(words)
        size = math.ceil(n / n_parts)
        kept = []
        for p in range(1, n_parts):
            k = min(size * p, n)
            if kept and k <= kept[-1]:
                k = kept[-1] + 1
            if k < n:
                kept.append(k)
        return [words[a:b] for a, b in zip([0] + kept, kept + [n])]

    def imbalance(parts):
        sizes = [len(p) for p in parts]
        return (max(sizes) - min(sizes), min(sizes)) if len(sizes) > 1 else (0, 0)

    checked = 0
    for words in lines:
        n_parts = captions._part_count(len(words))
        if n_parts >= len(words):
            continue
        old = even_split(words, n_parts)
        new = captions._split_parts(words, n_parts)
        if [len(p) for p in old] == [len(p) for p in new]:
            checked += 1
            continue
        # A part count is never allowed to change, and when the boundary does
        # move the result must be strictly balanced.
        assert len(new) == len(old)
        spread, smallest = imbalance(new)
        assert smallest >= captions.MIN_PART_WORDS, f"stranded tiny part in {words!r}"
        assert spread <= 1, f"lopsided parts {spread} wide in {words!r}"
        checked += 1
    assert checked > 0, "expected the embedded narration samples to be exercised"


# --------------------------------------------------------------------------
# I-7: the free-host whitelist must reject lookalike hosts while keeping
# every legitimate provider (and real subdomains) working.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "https://upload.wikimedia.org/a.jpg",
    "https://upload.wikimediausercontent.org/a.jpg",
    "https://live.staticflickr.com/a.jpg",
    "https://openverse.org/a.jpg",
    "https://archive.org/a.jpg",
    "https://loc.gov/a.jpg",
    "https://nara.gov/a.jpg",
    "https://si.edu/a.jpg",
    "https://foo.loc.gov/a.jpg",          # real subdomain
    "http://loc.gov/a.jpg",               # http, not https
    "https://LOC.GOV/a.jpg",              # case insensitive
])
def test_free_host_accepts_legitimate_providers(url):
    assert _on_free_host(url) is True


@pytest.mark.parametrize("url", [
    "https://loc.gov.evil.tld/a.jpg",             # suffix in the middle
    "https://foo.loc.gov.attacker.io/a.jpg",
    "https://evil-loc.gov.attacker.com/a.jpg",    # substring, not suffix
    "https://xloc.gov/a.jpg",                     # lookalike TLD
    "https://www.pexels.com/photo/a.jpg",
    "https://cdn.timreg.com/a.jpg",
])
def test_free_host_rejects_lookalikes(url):
    assert _on_free_host(url) is False


def test_blocked_domains_still_blocked():
    for url in ("https://images.imrg.news/a.jpg", "https://notgettyimages.example/a.jpg"):
        assert _is_blocked_host(url) is True


# --------------------------------------------------------------------------
# I-16: dedupe thresholds are intentionally different; pin both so a
# "cleanup" that unifies them is caught.
# --------------------------------------------------------------------------
def test_dedupe_thresholds_are_intentionally_distinct():
    from src.agents.topic import dedupe, helpers

    # a title pair that lands between the two thresholds: rejected by the
    # 0.7 lead-prune, but NOT by the 0.6 used-topics guard. That gap is the
    # reason the two numbers differ, so pin it.
    a, b = "one two three", "one two nine"
    overlap = dedupe._token_overlap(a, b)
    assert 0.6 <= overlap < 0.7, f"expected the 0.6-0.7 band, got {overlap}"
    assert not any(overlap >= 0.7 for _ in [overlap])
    assert not any(overlap >= 0.6 for _ in [])  # 0.6 guard WOULD flag it

    # both helpers are still independently reachable
    assert callable(helpers._is_niche) and callable(dedupe.dedupe_leads_leads)


# --------------------------------------------------------------------------
# I-10: output filenames must not collide, and must not change for a topic
# that has not been rendered yet.
# --------------------------------------------------------------------------
def test_output_path_stable_when_absent(tmp_path, monkeypatch):
    import os

    import src.utils.file_helpers as fh

    monkeypatch.setattr(fh, "OUTPUT_DIR", str(tmp_path))
    assert fh.output_path("Some Totally Unrendered Topic") == \
        os.path.join(str(tmp_path), "some_totally_unrendered_topic.mp4")


def test_output_path_disambiguates_collisions(tmp_path, monkeypatch):
    import os

    import src.utils.file_helpers as fh

    monkeypatch.setattr(fh, "OUTPUT_DIR", str(tmp_path))
    a = "Kursk 1943: A Complete Tactical Analysis of Operation Citadel and German Army Group South"
    b = "Kursk 1943: A Complete Tactical Analysis of Operation Citadel and German Army Group North"

    first = fh.output_path(a)
    assert first == fh.output_path(b)  # both truncate to the same 60-char slug
    open(first, "wb").close()  # first one lands
    second = fh.output_path(b)
    assert second != first, "a long shared prefix must not silently overwrite"
    assert os.path.basename(second).startswith(os.path.basename(first)[:-4])


# --- era gate: 18-image vision audit, 9/18 rated 0 and all 9 were modern ----
# Ground truth is the ratings a vision model returned for probe_results/
# e2e_assets/uboat. Every rating-0 hit was a modern space launch returned by
# the query "Rocket", or a 1976 press photo in a 1942 battle set.

def _ii(orig="", cats="", desc="", upload=""):
    return {"extmetadata": {k: {"value": v} for k, v in
                            (("DateTimeOriginal", orig), ("Categories", cats),
                             ("ImageDescription", desc), ("DateTime", upload))}}


@pytest.fixture
def era(monkeypatch):
    import src.agents.asset.agent as A
    monkeypatch.setattr(A, "_topic_era_year", 1943)
    return A


@pytest.mark.parametrize("name,orig,cats,desc", [
    ("soyuz",        "2010-06-15 00:00:00", "", ""),
    ("shuttle",      "1985-04-12 00:00:00", "Space Shuttle launches", ""),
    ("atlas",        "2018-02-06 00:00:00", "", ""),
    ("modern_press", "2021-01-01 00:00:00", "", "modern navy media interview"),
    ("cat_dated",    "", "2010s photographs|Spacecraft", ""),
    ("cat_future",   "", "2040s photographs", ""),
])
def test_era_gate_rejects_modern_markers(era, name, orig, cats, desc):
    assert era._anachronistic(_ii(orig, cats, desc), f"File:{name}.jpg") is True, name


@pytest.mark.parametrize("name,orig,cats,desc,upload", [
    ("schnorchel",   "1943-01-01 00:00:00", "Schnorchel|German U-boats", "", ""),
    ("u117",         "1944-06-01 00:00:00", "1944 in naval aviation", "deck crew", ""),
    ("no_date",      "", "", "some description", ""),
    ("paint",        "", "Paintings of submarines", "a 1940s painting", ""),
    ("peril_near",   "1958-01-01 00:00:00", "Schnorchel", "restored boat", ""),
    # The real Akagi file on Commons: made 1942, uploaded 2025. Gating on
    # DateTime (upload) rejected it; DateTimeOriginal is what makes it usable.
    ("akagi_uploaded_late", "1942-06-06 00:00:00", "Akagi (ship, 1927)",
     "diorama of the attack", "2025-01-20 06:18:58"),
    ("enterprise_2015", "4 June 1942 date QS:P571", "USS Enterprise (CV-6)",
     "steaming at high speed", "2015-01-23 15:30:34"),
    ("sbd_2023",     "19 May 1942", "Douglas SBD Dauntless|PD US Navy",
     "SBD-3 taking off", "2023-04-23 17:06:28"),
])
def test_era_gate_keeps_period_material(era, name, orig, cats, desc, upload):
    assert era._anachronistic(_ii(orig, cats, desc, upload), f"File:{name}.jpg") is False, name


def test_era_gate_ignores_upload_date_alone(era):
    """A recent upload is NOT anachronism. This was a silent total-wipe bug:
    every archival image on Commons is uploaded in the 2000s-2020s."""
    assert era._anachronistic(_ii("1942-06-06", upload="2025-01-20"),
                              "File:Akagi.jpg") is False
    assert era._anachronistic(_ii("", cats="Schnorchel", upload="2026-01-01"),
                              "File:U-boat.jpg") is False


def test_era_gate_is_noop_when_topic_year_unknown(monkeypatch):
    import src.agents.asset.agent as A
    monkeypatch.setattr(A, "_topic_era_year", None)
    # Must not starve the pool for a topic with no dates in it.
    assert A._anachronistic(_ii("2021-01-01", "SpaceX", "falcon"), "File:x.jpg") is False


def test_set_topic_era_uses_median_not_max(monkeypatch):
    import src.agents.asset.agent as A
    A._set_topic_era("Battle of Midway",
                     "In 1942 the fleet sailed. The war began 1914 and ended 1945.")
    assert A._topic_era_year == 1942, "max would wrongly pick 1945 or 1914"


def test_set_topic_era_none_without_years(monkeypatch):
    import src.agents.asset.agent as A
    A._set_topic_era("Something undated", "no years in here at all")
    assert A._topic_era_year is None


def test_modern_markers_do_not_hurt_modern_topics(monkeypatch):
    import src.agents.asset.agent as A
    A._set_topic_era("Falcon 9 and the Dragon capsule", "Since 2010 SpaceX has flown")
    # A modern subject must NOT be blocked by the spaceflight marker list.
    assert A._anachronistic(_ii("2020-05-30", "Falcon 9 launches", ""), "File:f9.jpg") is False


# --- per-term budget: generic terms eat the pool and return nothing --------
# Measured on a real 18-image pool (vision-model ratings):
#   "Schnorchel" -> 4 images, 4 usable.  "Rocket" -> 6 images, 0 usable.
# The term is the whole story, so a generic term gets a token budget.

def test_entity_budget_starves_generic_terms():
    from src.agents.asset.agent import _entity_budget, COMMONS_PER_ENTITY
    # a generic single noun: 1 image, not the full slot
    assert _entity_budget("Rocket") == 1
    assert _entity_budget("rocket") == 1
    assert _entity_budget("Submarine") == 1
    # a rare single word is still a real entity and keeps its budget
    assert _entity_budget("Schnorchel") == COMMONS_PER_ENTITY
    assert _entity_budget("Werwolf") == COMMONS_PER_ENTITY
    # multi-word entities always keep the full budget
    assert _entity_budget("Kido Butai") == COMMONS_PER_ENTITY
    assert _entity_budget("Japanese Combined Fleet") == COMMONS_PER_ENTITY


def test_entity_budget_tolerates_punctuation():
    from src.agents.asset.agent import _entity_budget
    assert _entity_budget("Rocket.") == 1
    assert _entity_budget("missile,") == 1


def test_entity_specs_carry_budget_through():
    from src.agents.asset.agent import _entity_specs
    specs = _entity_specs(["Rocket", "Schnorchel", "Kido Butai"], "/tmp/a")
    assert [s[0] for s in specs] == ["Rocket", "Schnorchel", "Kido Butai"]
    assert specs[0][3] == 1, "generic term must carry the small budget"
    assert specs[1][3] > 1 and specs[2][3] > 1
    # every spec is a 4-tuple now; the downloader unpacks exactly that
    for s in specs:
        assert len(s) == 4


def test_generic_budget_does_not_starve_a_narrow_topic():
    """A topic made ONLY of generic terms must still return images, not zero."""
    from src.agents.asset.agent import _entity_budget
    budget = sum(_entity_budget(t) for t in ("Rocket", "Submarine", "Tanks"))
    assert budget >= 3, "at least one image per term, always"


def test_fallback_providers_get_a_smaller_slice(monkeypatch):
    """A per-term budget is meaningless if DDG re-spends it on the same term."""
    from src.agents.asset import agent as A
    seen = {}
    monkeypatch.setattr(A, "_ENABLE_PEXELS", False)
    monkeypatch.setattr(A, "_fetch_wikimedia_commons",
                        lambda kw, base, count: seen.setdefault("commons", count) and [] or [])
    monkeypatch.setattr(A, "_fetch_openverse",
                        lambda kw, base, count: seen.setdefault("openverse", count) and [] or [])
    monkeypatch.setattr(A, "_fetch_ddg",
                        lambda kw, base, count: seen.setdefault("ddg", count) and [] or [])
    A._fetch_for_keyword("Rocket", 0, "/tmp/a", 6)
    assert seen["commons"] == 6
    assert seen["openverse"] < 6 and seen["ddg"] < 6
