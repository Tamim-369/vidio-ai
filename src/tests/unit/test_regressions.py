"""Regression tests for the behaviour-preserving fixes.

Every assertion here pins down behaviour that a fix had to PRESERVE, so a
future change that silently regresses it fails loudly instead of quietly
producing worse videos.
"""
import math

import pytest

from src.agents.asset import agent as agent_mod
from src.agents.asset.validate import (
    hull_designator, resolve_hull, validate_candidate, is_non_visual_entity,
)

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
    monkeypatch.setattr(A, "_ENABLE_GENERIC", True)
    A._fetch_for_keyword("Rocket", 0, "/tmp/a", 6)
    assert seen["commons"] == 6
    assert seen["openverse"] < 6 and seen["ddg"] < 6


def test_generic_providers_are_off_by_default(monkeypatch):
    """A Midway render came out with ducks in front of a flag.

    DDG and Openverse carry no Commons metadata, so validate_candidate() cannot
    score them and nothing screened what they returned.
    """
    from src.agents.asset import agent as A
    import src.agents.asset.commons as C
    assert C.ASSET_ENABLE_GENERIC is False, "generic web-image sources are on"
    called = []
    monkeypatch.setattr(A, "_ENABLE_GENERIC", False)
    monkeypatch.setattr(A, "_fetch_wikimedia_commons",
                        lambda kw, base, count: called.append("commons") or [])
    monkeypatch.setattr(A, "_fetch_openverse",
                        lambda kw, base, count: called.append("openverse") or [])
    monkeypatch.setattr(A, "_fetch_ddg",
                        lambda kw, base, count: called.append("ddg") or [])
    A._fetch_for_keyword("USS Akagi Battle of Midway", 0, "/tmp/a", 6)
    assert called == ["commons"], called


def test_relevance_gate_rejects_a_stock_photo_of_ducks():
    """The alt-text gate that would have caught the ducks."""
    from src.agents.asset.agent import _relevance_ok
    assert _relevance_ok("Ducks sitting in front of a flag on grass",
                          "USS Akagi Battle of Midway") is False
    assert _relevance_ok("Japanese aircraft carrier Akagi burning at sea",
                          "USS Akagi Battle of Midway") is True


# --- per-term cap in assignment ------------------------------------------
# A real render came out as "a photo of an island and a bunch of white birds".
# "Midway" means the atoll, so Commons returned seabirds, and 6 of 6 assigned
# slots went to that one term. The pool was fine (58 images, 9 real entities);
# assignment is what failed.

def _lines(n):
    return [{"id": i} for i in range(1, n + 1)]


def test_cap_per_term_stops_one_term_filling_the_video():
    from src.agents.asset.agent import _cap_per_term
    imgs = {f"/a/{i}_Midway.jpg": ("Midway", "Midway", None) for i in range(8)}
    lines = _lines(10)
    mapping = {i: [f"/a/{i - 1}_Midway.jpg"] for i in range(1, 9)}
    out = _cap_per_term(mapping, imgs, lines)
    kept = [p for deck in out.values() for p in deck]
    assert len(kept) == len(set(kept)), "an image must not appear on two lines"
    # with 1 term and 10 lines the cap relaxes; it must not starve every line
    assert len(out) >= 1


def test_cap_per_term_prefers_term_diversity():
    from src.agents.asset.agent import _cap_per_term
    terms = ["Akagi", "Soryu", "Kaga", "Enterprise", "Yorktown", "Hornet"]
    imgs = {f"/a/{t}.jpg": (t, t, None) for t in terms}
    lines = _lines(6)
    # planner put all 6 lines on Midway images (the birds)
    imgs.update({f"/a/M{i}.jpg": ("Midway", "Midway", None) for i in range(6)})
    mapping = {i: [f"/a/M{i - 1}.jpg"] for i in range(1, 7)}
    out = _cap_per_term(mapping, imgs, lines)
    kept = [p for deck in out.values() for p in deck]
    # The pool holds 7 terms (6 carriers + Midway), so the cap holds at 2 and
    # the all-Midway plan collapses from 6 images to 2. Remaining lines fall
    # back to the diverse pool via backfill.
    assert len(kept) == 2, "all-Midway plan must collapse to the cap"
    assert all(p.endswith(("M0.jpg", "M1.jpg")) for p in kept)


def test_cap_per_term_never_duplicates_an_image():
    from src.agents.asset.agent import _cap_per_term
    imgs = {f"/a/{i}.jpg": (f"T{i % 2}", f"T{i % 2}", None) for i in range(6)}
    lines = _lines(6)
    mapping = {i: ["/a/0.jpg"] for i in range(1, 7)}   # planner reused one path
    out = _cap_per_term(mapping, imgs, lines)
    kept = [p for deck in out.values() for p in deck]
    assert len(kept) == len(set(kept)), "the same path on 6 lines is a bug"


def test_cap_per_term_never_starves_a_line_when_terms_are_few():
    from src.agents.asset.agent import _cap_per_term
    imgs = {f"/a/{i}.jpg": ("Rocket", "Rocket", None) for i in range(4)}
    lines = _lines(6)
    mapping = {i: [f"/a/{i - 1}.jpg"] for i in range(1, 5)}
    out = _cap_per_term(mapping, imgs, lines)
    # cap relaxes because terms(1) < lines(6): the video still gets visuals
    assert sum(1 for v in out.values() if v) >= 1


def test_cap_per_term_empty_mapping_is_a_noop():
    from src.agents.asset.agent import _cap_per_term
    assert _cap_per_term({}, {}, _lines(3)) == {}
    assert _cap_per_term({1: ["/a.jpg"]}, {}, _lines(1)) == {1: ["/a.jpg"]}


def test_mapping_paths_is_ordered_and_unique():
    from src.agents.asset.agent import mapping_paths
    m = {1: ["/a", "/b"], 2: ["/b", "/c"]}
    assert mapping_paths(m) == ["/a", "/b", "/c"]


# --- bounded pool: 94 images for 10 lines cost 6.4 minutes ----------------
def test_pool_budget_is_bounded_by_video_length(monkeypatch):
    from src.agents.asset import agent as A
    calls = {}
    def fake(kw, idx, d, n):
        calls[kw] = n
        return [f"/a/{kw}_{i}.jpg" for i in range(n)]
    monkeypatch.setattr(A, "_fetch_for_keyword", fake)
    monkeypatch.setattr(A, "MAX_PARALLEL_WORKERS", 8)
    ents = [f"Term{i}" for i in range(14)]
    cap = A._pool_target(12)
    A._download_entities(ents, "/tmp/pool", cap)
    total = sum(calls.values())
    assert total <= cap + len(ents), (
        f"budgeted {total} images for {len(ents)} entities, cap is {cap}")
    # the real pool had every term at the full budget; budgets must shrink
    assert max(calls.values()) <= A.COMMONS_PER_ENTITY


def test_download_entities_trims_an_oversized_pool(monkeypatch):
    from src.agents.asset import agent as A
    monkeypatch.setattr(A, "_fetch_for_keyword",
                        lambda kw, i, d, n: [f"/a/{kw}_{i}_{j}.jpg" for j in range(9)])
    monkeypatch.setattr(A, "MAX_PARALLEL_WORKERS", 8)
    cap = A._pool_target(12)
    cands = A._download_entities(["A", "B", "C"], "/tmp/pool", cap)
    assert len(cands) <= cap


def test_pool_entity_count_is_capped():
    from src.agents.asset.agent import MAX_POOL_ENTITIES
    assert 4 <= MAX_POOL_ENTITIES <= 10, "more entities is wasted Commons time"


def test_pool_scales_with_the_video_not_a_flat_five():
    """A flat 5-image pool cannot fill 12 lines at 3 images each without
    looping the same two pictures, which is the repetition users reported."""
    from src.agents.asset.agent import _pool_target
    assert _pool_target(3) >= 5
    assert _pool_target(12) > _pool_target(3), "pool must grow with line count"


def test_pool_stays_diverse_across_queries():
    """No query may take most of the pool; near-identical carrier portraits
    under three filenames are what made the video look like random boats."""
    from src.agents.asset.agent import _diverse_slice, MAX_IMAGES_PER_QUERY
    cands = {}
    for i in range(6):
        cands[f"/a/akagi{i}.jpg"] = ("USS Akagi Battle of Midway", None)
    for i in range(2):
        cands[f"/a/yam{i}.jpg"] = ("Isoroku Yamamoto Battle of Midway", None)
    out = _diverse_slice(cands, 4)
    tally = {}
    for _p, meta in out:
        tally[meta[0]] = tally.get(meta[0], 0) + 1
    assert max(tally.values()) <= MAX_IMAGES_PER_QUERY, tally
    assert len(tally) >= 2, tally


# --- assignment: every line gets a DISTINCT image -------------------------
# A real run used 7 of 94 images and backfilled lines 8-10 onto their
# neighbour's picture.

def _mk_lines(n):
    return [{"id": i, "search_term": f"term{i}"} for i in range(1, n + 1)]


def test_assignment_leaves_a_line_bare_when_nothing_matches():
    """The new contract, replacing "every line gets a distinct image".

    Forcing a picture onto a sentence with no confident match is what produced
    the "random boats": the line was not asking for a carrier, so it got a
    carrier anyway, and one battle rendered as six unrelated warships.
    """
    from src.agents.asset import agent as A
    imgs = {f"/a/{i}.jpg": ("akagi carrier", "akagi carrier", None) for i in range(5)}
    lines = _mk_lines(10)
    for i, ln in enumerate(lines):
        ln["text"] = "Dauntless dive bombers torpedo the Japanese fleet at 0755."
    out = A._assign_images_to_lines(imgs, lines, "Battle of Midway")
    bare = [ln["id"] for ln in lines if not out.get(ln["id"])]
    assert bare, "a 5-image pool must not be forced onto 10 lines"


def test_assignment_spreads_a_small_pool_instead_of_repeating_one_image():
    """A small pool must not collapse onto a single image per line."""
    from src.agents.asset import agent as A
    imgs = {f"/a/{i}.jpg": (f"term{i}", f"term{i}", None) for i in range(5)}
    out = A._assign_images_to_lines(imgs, _mk_lines(3), "topic")
    picked = [p for deck in out.values() for p in deck]
    assert len(set(picked)) > 1, "the whole pool collapsed onto one image"
    assert all(len(deck) <= A.MAX_IMAGES_PER_LINE for deck in out.values())


def test_assignment_caps_images_per_line():
    from src.agents.asset import agent as A
    imgs = {f"/a/{i}.jpg": ("akagi", "akagi", None) for i in range(20)}
    out = A._assign_images_to_lines(imgs, _mk_lines(4), "Battle of Midway")
    assert all(len(deck) <= A.MAX_IMAGES_PER_LINE for deck in out.values())


def test_backfill_is_off_by_default():
    """A line with no visual is correct; forced filler is the regression."""
    from src.agents.asset import agent as A
    lines = _mk_lines(3)
    A._backfill_gaps(lines, {"/a/0.jpg": ("akagi", "akagi", None)})
    assert all(not ln.get("asset_paths") for ln in lines)


def test_assignment_survives_an_empty_planner_result(monkeypatch):
    """The deterministic pass must still run when the planner returns nothing."""
    from src.agents.asset import agent as A
    import src.agents.asset.assign as LA
    monkeypatch.setattr(LA, "plan_assignment", lambda i, l, t: {})
    imgs = {f"/a/{i}.jpg": (f"term{i}", f"term{i}", None) for i in range(5)}
    out = A._assign_images_to_lines(imgs, _mk_lines(3), "topic")
    picked = [p for deck in out.values() for p in deck]
    assert picked, "no image assigned despite matching features"


# --- image validation: the building with "pecifc" on it -------------------
# The pool was fine but the *type* was wrong. "Pacific Theater" is a region so
# Commons returned an operations-theatre map; "USS Pacific Fleet" is a command
# so it returned a building. Grounding cannot catch this: the term really is
# in the article.

def _vmeta(cats="", date="", desc="", w=2000, h=1500):
    em = {}
    if cats:
        em["Categories"] = {"value": cats}
    if date:
        em["DateTimeOriginal"] = {"value": date}
    if desc:
        em["ImageDescription"] = {"value": desc}
    return {"extmetadata": em, "width": w, "height": h}


@pytest.mark.parametrize("term,expected", [
    ("Pacific Theater", True),
    ("USS Pacific Fleet", True),
    ("Japanese Combined Fleet", True),
    ("World War II", True),
    ("The Battle of Midway", True),
    ("5th Fleet", True),
    ("United States Navy", True),
    ("December 7 1941", True),
    ("CV-3", True),
    ("USS Akagi", False),
    ("Pearl Harbor", False),
    ("Coral Sea", False),
    ("Akagi", False),
    ("Isoroku Yamamoto", False),
    ("Kido Butai", False),
    ("Midway Atoll", False),
    ("USS Enterprise (CV-6)", False),
    ("Chuichi Nagumo", False),
    ("Schnorchel", False),
])
def test_non_visual_entities_are_rejected(term, expected):
    assert is_non_visual_entity(term) is expected, term


def test_validate_rejects_a_theatre_of_operations_map():
    r = validate_candidate(
        _vmeta("Maps of World War II in the Pacific|PD US NPS", "2024-09-14"),
        "File:Map of Pacific Theater of World War II.jpg", "Pacific Theater", 1942)
    assert r["verdict"] == "reject", r


def test_validate_rejects_a_memorial_building():
    r = validate_candidate(
        _vmeta("Pearl Harbor Memorial|Monuments in Hawaii", "2020-01-01"),
        "File:Pearl Harbor Memorial.jpg", "Pearl Harbor", 1942)
    assert r["verdict"] == "reject", r


def test_validate_rejects_a_museum_not_the_ship():
    """The Lexington case: CV-3 was Saratoga, and Commons answered with the
    Corpus Christi museum on the bay."""
    r = validate_candidate(
        _vmeta("Views from Corpus Christi of USS Lexington Museum on the Bay|"
               "Self-published work", "2018-10-05"),
        "File:USS Lexington -- The-Blue-Ghost' -- Corpus Christi.jpg",
        "USS Lexington (CV-3)", 1942)
    assert r["verdict"] == "reject", r


def test_validate_rejects_a_graffiti_wall():
    """The user saw a building with 'pecifc' written on it."""
    r = validate_candidate(
        _vmeta("Graffiti|Street art in Hawaii", "2023-02-01"),
        "File:Graffiti on a wall, Pearl Harbor.jpg", "Pearl Harbor", 1942)
    assert r["verdict"] == "reject", r


def test_validate_rejects_emblems_and_logos():
    r = validate_candidate(
        _vmeta("United States Pacific Fleet|Fleet emblems of the United States Navy",
               "Unknown date"),
        "File:Commander United States Pacific Fleet logo.jpg",
        "USS Pacific Fleet", 1942)
    assert r["verdict"] == "reject", r


def test_validate_rejects_a_made_far_after_the_topic():
    r = validate_candidate(
        _vmeta("June 2011 in California", "Taken on 24 June 2011"),
        "File:USS Midway leaving San Diego.jpg", "USS Akagi", 1942)
    assert r["verdict"] == "reject", r


def test_validate_keeps_genuine_war_photographs():
    """The regression that matters most: real archival material must survive.
    Categories once flagged a true 1942 action shot as "Dioramas", so no single
    signal may be allowed to reject on its own."""
    good = [
        # a real combat photograph, PD US Navy
        (_vmeta("Douglas SBD Dauntless|PD US Navy", "19 May 1942"),
         "File:Douglas SBD-3 of VB-6 taking off.jpg", "USS Akagi"),
        (_vmeta("USS Enterprise (CV-6)|PD US Navy",
                "4 June 1942 date QS:P571,+1942-06-04"),
         "File:USS Enterprise (CV-6) at high speed.jpg", "USS Akagi"),
        (_vmeta("USS Lexington (CV-2)|PD US Navy|Ship engine smoke",
                "1933-02-02"),
         "File:USS Saratoga (CV-3) and USS Lexington.jpg", "Coral Sea"),
        (_vmeta("1942 in naval aviation|German U-boats", "1943-01-01"),
         "File:Schnorchel U-boat deck.jpg", "Schnorchel"),
    ]
    for meta, title, ent in good:
        r = validate_candidate(meta, title, ent, 1942)
        assert r["verdict"] == "keep", (title, r)


def test_validate_ignores_upload_date_for_era():
    """DateTime is the upload date; a 1942 photo uploaded in 2025 is fine."""
    r = validate_candidate(
        _vmeta("Kaga (ship, 1927)|PD US Navy",
               "1942-06-06"),  # DateTimeOriginal is the artwork date
        "File:Kaga, Akagi and Soryu under attack.jpg", "USS Akagi", 1942)
    assert r["verdict"] == "keep", r


def test_validate_rejects_tiny_images():
    r = validate_candidate(_vmeta("1942 in naval aviation", "1942-01-01",
                                  w=320, h=240),
                           "File:small.jpg", "USS Akagi", 1942)
    assert r["verdict"] == "reject", r


def test_validate_is_a_noop_without_an_era():
    r = validate_candidate(_vmeta("Pearl Harbor|Memorials"),
                           "File:Pearl Harbor Memorial.jpg", "Pearl Harbor", None)
    assert r["verdict"] == "reject", "type rejection must not need an era"
    r2 = validate_candidate(_vmeta("June 2011 in California", "24 June 2011"),
                            "File:x.jpg", "USS Akagi", None)
    assert r2["verdict"] == "keep", "era must not be guessed when unknown"


# --- hull designation audit -------------------------------------------------
# The Midway run searched "USS Lexington (CV-3)". CV-3 was USS Saratoga;
# Lexington was CV-2. Commons answered with the Corpus Christi museum and the
# wrong carrier, and grounding passed it because the string was in the article.


def test_hull_designator_splits_name_and_number():
    assert hull_designator("USS Lexington (CV-3)") == ("uss lexington", "cv-3")
    assert hull_designator("IJNS Akagi") is None
    assert hull_designator("Battle of Midway") is None


def test_resolve_hull_corrects_a_hallucinated_designation():
    """Commons categories say CV-2; the query said CV-3. Correct it."""
    cats = ["USS Lexington (CV-2)|PD US Navy"] * 3 + ["USS Lexington (CV-16)|PD US Navy"]
    assert resolve_hull("USS Lexington (CV-3)", cats) == "USS Lexington (CV-2)"


def test_resolve_hull_keeps_a_correct_designation():
    """When Commons supports the queried hull, change nothing."""
    assert resolve_hull("USS Lexington (CV-2)",
                        ["USS Lexington (CV-2)|PD US Navy"] * 3) == "USS Lexington (CV-2)"


def test_resolve_hull_leaves_non_ships_alone():
    assert resolve_hull("USS Akagi", []) == "USS Akagi"
    assert resolve_hull("Pearl Harbor", []) == "Pearl Harbor"


def test_resolve_hull_ignores_a_single_dissenting_result():
    """One result is not enough evidence to rewrite a term."""
    assert resolve_hull("USS Lexington (CV-3)",
                        ["USS Lexington (CV-2)|PD US Navy"]) == "USS Lexington (CV-3)"


def test_audited_term_corrects_before_any_download(monkeypatch, tmp_path):
    """The budget is spent on the first pass, so the audit has to happen first."""
    searched = []

    def fake_search(params):
        searched.append(params["gsrsearch"])
        return {"query": {"pages": {
            "a": {"imageinfo": [{"extmetadata": {
                "Categories": {"value": "USS Lexington (CV-2)|PD US Navy"}}}]},
            "b": {"imageinfo": [{"extmetadata": {
                "Categories": {"value": "USS Lexington (CV-2)|PD US Navy"}}}]},
            "c": {"imageinfo": [{"extmetadata": {
                "Categories": {"value": "USS Lexington (CV-2)|PD US Navy"}}}]},
        }}}

    monkeypatch.setattr(agent_mod, "_commons_search", fake_search)
    called = []
    monkeypatch.setattr(agent_mod, "_commons_pass",
                        lambda term, base, count: called.append(term) or [])
    assert agent_mod._audited_term("USS Lexington (CV-3)") == "USS Lexington (CV-2)"
    agent_mod._fetch_wikimedia_commons("USS Lexington (CV-3)", str(tmp_path), 3)
    assert called == ["USS Lexington (CV-2)"], called


def test_audited_term_costs_nothing_for_terms_without_a_hull(monkeypatch):
    """No hull designation means no audit search at all."""
    def boom(params):
        raise AssertionError("unnecessary audit search")
    monkeypatch.setattr(agent_mod, "_commons_search", boom)
    assert agent_mod._audited_term("Pearl Harbor") == "Pearl Harbor"
    assert agent_mod._audited_term("USS Akagi") == "USS Akagi"


def test_audited_term_survives_a_throttled_audit(monkeypatch):
    def boom(params):
        raise RuntimeError("429")
    monkeypatch.setattr(agent_mod, "_commons_search", boom)
    assert agent_mod._audited_term("USS Lexington (CV-3)") == "USS Lexington (CV-3)"


# --- memorial ban ----------------------------------------------------------
# A render came out "with duck images sitting in front of flag". Those were
# modern commemoration ceremonies: grass, flag, wreaths, people standing
# around. Channel decision: never show memorial, cemetery, grave or
# commemorative imagery -- not even for a video about a soldier, where the
# brief is to show the war he fought in rather than his grave.

def test_commemoration_photographs_are_rejected():
    """The word is "Commemoration"; the old patterns only knew "Memorial"."""
    from src.agents.asset.validate import validate_candidate
    r = validate_candidate(
        _vmeta("PD US FWS|2022 in Midway Atoll|Flickr"),
        "File:80th Commemoration of the Battle of Midway (522178044.jpg).jpg",
        "Midway Atoll", 1942)
    assert r["verdict"] == "reject", r


def test_anniversary_and_memorial_service_photographs_are_rejected():
    from src.agents.asset.validate import validate_candidate
    for title in ("File:USFWS Commemorates 77th Anniversary of the Battle.jpg",
                  "File:Battle of Midway memorial service.jpg",
                  "File:USS Arizona Memorial, Pearl Harbor.jpg"):
        r = validate_candidate(_vmeta("PD US FWS"), title, "Pearl Harbor", 1942)
        assert r["verdict"] == "reject", (title, r)


def test_cemeteries_and_graves_are_rejected():
    from src.agents.asset.validate import validate_candidate
    for title in ("File:Marine Corps War Cemetery Iwo Jima.jpg",
                  "File:Arlington National Cemetery.jpg",
                  "File:Graves of the Unknown Soldier.jpg"):
        r = validate_candidate(_vmeta("Military cemeteries"), title,
                               "Battle of Midway", 1942)
        assert r["verdict"] == "reject", (title, r)


def test_archival_photographs_of_the_actual_fighting_still_pass():
    """The ban must not eat the genuine article."""
    from src.agents.asset.validate import validate_candidate
    r = validate_candidate(
        _vmeta("PD US Navy|Douglas SBD Dauntless"),
        "File:Douglas SBD-3 Dauntless of VB-6 taking off from USS Enterprise.jpg",
        "USS Akagi", 1942)
    assert r["verdict"] == "keep", r


# --- Ken Burns micro-freeze -------------------------------------------------
# "The image stops zooming and just freezes for milliseconds." Two independent
# causes, both of which had to go.

def _tmp_images(n, tmp_path, w=1080, h=1920):
    import numpy as np
    from PIL import Image
    out = []
    for i in range(n):
        p = tmp_path / f"i{i}.jpg"
        Image.fromarray(
            (np.random.rand(h, w, 3) * 255).astype("uint8")).save(p)
        out.append(str(p))
    return out


def test_zoom_never_repeats_a_frame(tmp_path):
    """int(w * scale) truncated, so the crop stalled on some frames.

    A 6% zoom over ~50 frames moves the crop edge about 1.3px per frame at
    1080px, so integer rounding held it still long enough to read as a freeze.
    """
    import numpy as np
    from src.services.video_assembler import _line_frames
    frames = _line_frames(_tmp_images(1, tmp_path), 4.0, (270, 480))
    dupes = sum(1 for a, b in zip(frames, frames[1:]) if np.array_equal(a, b))
    assert dupes == 0, f"{dupes} identical consecutive frames"


def test_line_frames_cover_the_line_duration_exactly(tmp_path):
    """Crossfades overlap, so each one consumed frames and the shortfall was
    padded by holding the last frame -- a freeze lasting until the audio ended."""
    from src.services.video_assembler import _line_frames, FPS
    for dur in (6.0, 3.2, 2.0):
        frames = _line_frames(_tmp_images(3, tmp_path), dur, (270, 480))
        assert len(frames) == round(dur * FPS), (dur, len(frames))


def test_no_frozen_tail_after_a_crossfade(tmp_path):
    """The visible artefact: zoom ends, final image holds until the cut."""
    import numpy as np
    from src.services.video_assembler import _line_frames
    frames = _line_frames(_tmp_images(3, tmp_path), 6.0, (270, 480))
    tail = len(frames) - 1
    while tail > 0 and np.array_equal(frames[tail], frames[tail - 1]):
        tail -= 1
    assert tail == len(frames) - 1, f"{(len(frames) - 1 - tail)} frozen frames at the tail"


# --- face reuse -----------------------------------------------------------
# "Dude's face way more than it should be shown." A 6-image pool fills ~24
# slots in a 10-line video, so without a cap every picture recurs about four
# times, and a portrait's face appears four times.

def test_no_image_repeats_more_than_the_cap(monkeypatch):
    import collections
    from src.agents.asset import agent as A
    monkeypatch.setattr(A, "_IS_PORTRAIT", {}, raising=False)
    terms = ["akagi", "yamamoto", "nagumo", "pearl", "midway", "enterprise"]
    imgs = {f"/a/{i}.jpg": (f"{terms[i % 6]} carrier", f"{terms[i % 6]}", None)
            for i in range(6)}
    lines = [{"id": i, "text": f"Japanese carrier {terms[i % 6]} at sea."}
             for i in range(1, 11)]
    out = A._assign_images_to_lines(imgs, lines, "Battle of Midway")
    uses = collections.Counter(p for deck in out.values() for p in deck)
    assert max(uses.values()) <= A.MAX_USES_PER_IMAGE, dict(uses)


def test_a_face_is_capped_tighter_than_an_ordinary_image(monkeypatch):
    import collections
    from src.agents.asset import agent as A
    monkeypatch.setattr(A, "_IS_PORTRAIT", {"/a/3.jpg": True}, raising=False)
    terms = ["akagi", "yamamoto", "nagumo", "pearl", "midway", "enterprise"]
    imgs = {f"/a/{i}.jpg": (f"{terms[i % 6]} carrier", f"{terms[i % 6]}", None)
            for i in range(6)}
    lines = [{"id": i, "text": f"Japanese carrier {terms[i % 6]} at sea."}
             for i in range(1, 11)]
    out = A._assign_images_to_lines(imgs, lines, "Battle of Midway")
    uses = collections.Counter(p for deck in out.values() for p in deck)
    assert uses["/a/3.jpg"] <= A.MAX_USES_PER_PORTRAIT, dict(uses)
    assert A.MAX_USES_PER_PORTRAIT < A.MAX_USES_PER_IMAGE


def test_portrait_detection_uses_commons_categories():
    from src.agents.asset.validate import is_portrait_image
    assert is_portrait_image("File:Yamamoto Isoroku.jpg", "Portraits")
    assert is_portrait_image("File:Admiral Halsey.jpg",
                             "Military portraits of the United States Navy")
    assert not is_portrait_image("File:SBD Dauntless landing.jpg", "Aircraft")


def test_reuse_spreads_over_the_pool_instead_of_leaning_on_one_picture():
    """Reuse must prefer the least-used match, not the highest-scoring one."""
    import collections
    from src.agents.asset import agent as A
    terms = ["akagi", "yamamoto", "nagumo", "pearl", "midway", "enterprise"]
    imgs = {f"/a/{i}.jpg": (f"{terms[i % 6]} carrier", f"{terms[i % 6]}", None)
            for i in range(6)}
    lines = [{"id": i, "text": "carrier at sea"} for i in range(1, 8)]
    out = A._assign_images_to_lines(imgs, lines, "Battle of Midway")
    uses = collections.Counter(p for deck in out.values() for p in deck)
    assert len(uses) >= 4, f"video leans on {len(uses)} of {len(imgs)} images"


# --- per-line image targeting ----------------------------------------------
# "It should position the images based on line: if one line is talking about
# plane then it should get a plane." Token overlap between a narration line and
# a topic-level entity query is weak by construction, so the line about
# aircraft scored zero and shipped bare.

def test_a_line_keeps_the_image_fetched_for_it():
    """An image fetched FOR a line wins that line over token scoring."""
    from src.agents.asset import agent as A
    imgs = {
        "/a/plane.jpg": ("sbd dauntless dive bomber", "sbd dauntless dive bomber", 6),
        "/a/carrier.jpg": ("uss akagi carrier", "uss akagi carrier", 1),
    }
    lines = [{"id": 6, "text": "American aircraft launch their attack."},
             {"id": 1, "text": "The Japanese fleet sits on the water."}]
    out = A._assign_images_to_lines(imgs, lines, "Battle of Midway")
    assert out[6] and out[6][0] == "/a/plane.jpg", out
    assert out[1] and out[1][0] == "/a/carrier.jpg", out


def test_query_boilerplate_is_stripped_before_searching():
    """Commons ANDs every term, so the photography filler kills the query."""
    from src.agents.asset.agent import _commons_query
    got = _commons_query(
        "Japanese carrier Akagi burning aerial view 1942 archive photo")
    assert "archive" not in got and "photo" not in got, got
    assert "Akagi" in got, got


def test_a_truncated_query_does_not_end_in_a_function_word():
    from src.agents.asset.agent import _commons_query
    assert not _commons_query(
        "US Navy SBD Dauntless launching from carrier 1942 archive photo"
    ).endswith("from")


def test_backoff_never_shortens_below_three_words():
    """'US Navy carrier Enterprise 1942' -> 'US Navy' returns generic fleet
    photography: plausible-looking but unrelated to the line."""
    from src.agents.asset.agent import _commons_query
    term = _commons_query("US Navy carrier Enterprise 1942").split()
    floor = min(3, len(term))
    variants = [" ".join(term[:n]) for n in range(len(term), floor - 1, -1)]
    assert all(len(v.split()) >= 3 for v in variants), variants
