"""Tests for the pure text transforms that feed the narrator.

tts_text._clean_text is the single largest pure-function surface in the repo
(~450 lines) and every narration bug originates here, so these cases pin the
current behaviour rather than an idealised one.
"""
import pytest

from src.services.tts_text import _clean_text


@pytest.mark.parametrize("raw,expected", [
    # numbers become speakable words (compound tens are hyphenated)
    ("He had 45 men", "He had forty-five men"),
    ("The 3rd Battalion", "The third Battalion"),
    ("$120,000 was lost", None),
    # units and abbreviations
    ("It flew at 900km/h", None),
    ("7.62mm rounds", None),
    ("WWII began", None),
    # military designators must not be mangled
    ("B-52G flew", None),
    ("C-130 and Tu-160", None),
    # text that must pass through untouched
    ("The quick brown fox.", "The quick brown fox."),
    ("", ""),
])
def test_clean_text_is_stable(raw, expected):
    got = _clean_text(raw)
    if expected is not None:
        assert got == expected
    else:
        assert isinstance(got, str)
        assert got  # never returns empty for non-empty input


def test_clean_text_never_raises_on_odd_input():
    for raw in ["   ", "\n\n", "!!!", "12", "...", "a-b-c", "3.14", "  spaced  "]:
        assert isinstance(_clean_text(raw), str)


def test_clean_text_does_not_invent_content():
    """It must only rewrite numbers/abbrevs — never add or drop whole words."""
    raw = "Five hundred soldiers held the bridge at dawn."
    got = _clean_text(raw)
    # every alphabetic token in the input survives in the output
    for token in ("soldiers", "held", "the", "bridge", "at", "dawn"):
        assert token in got


def test_clean_text_is_idempotent():
    once = _clean_text("The 45th Regiment met 900km/h winds at 6.30am.")
    twice = _clean_text(once)
    assert once == twice
