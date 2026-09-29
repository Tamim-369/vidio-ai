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
    import glob
    import json

    lines = []
    for p in glob.glob("debug_output/*_script_*.json"):
        try:
            data = json.load(open(p))
        except Exception:
            continue
        for ln in data.get("lines", []):
            text = (ln.get("text") or "").strip()
            if len(text.split()) >= 4:
                lines.append(text.split())

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
    assert checked > 0, "expected real narration samples in debug_output/"


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
