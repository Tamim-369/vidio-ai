"""Characterization tests for TTS text normalization.

These lock in the EXACT current behaviour of ``tts_text._clean_text``, the
function the TTS path uses to turn narration into speakable text. They were
written by probing the real implementation, not from assumptions: the
normalizer spells numbers out, expands abbreviations letter-by-letter, and
deliberately leaves brackets and parentheses untouched (caption text and
stage directions are stripped upstream).

If a refactor changes any of these outputs, these tests fail. That is the
point: the restructuring mandate is that behaviour does not change.
"""
from __future__ import annotations

import pytest

from src.services.tts_text import _clean_text


class TestWhitespace:
    def test_collapses_whitespace_and_strips(self):
        assert _clean_text("  hello   world  ") == "hello world"

    def test_empty_is_safe(self):
        assert _clean_text("") == ""

    def test_none_raises(self):
        # Documents actual current behaviour: callers must pass a string.
        # (tts.py guards with `text or ""` before calling, so this is
        # unreachable in the pipeline.) If a future refactor makes this
        # lenient, that is an improvement, not a regression.
        with pytest.raises(TypeError):
            _clean_text(None)


class TestNumbersSpelledOut:
    """The engine reads digits unreliably, so every number becomes words."""

    def test_whole_number(self):
        assert _clean_text("Up 40 percent.") == "Up forty percent."

    def test_dollar_amount(self):
        assert _clean_text("It cost $5 million.") == "It cost five dollars million."

    def test_thousands_separator_is_read_as_hundreds(self):
        # "1,200" -> "twelve hundred" (spoken form, not "one thousand two hundred")
        assert _clean_text("1,200 troops died.") == "twelve hundred troops died."

    def test_year_becomes_spoken(self):
        assert _clean_text("In 1944, he left.") == "In nineteen forty-four, he left."

    def test_keeps_trailing_punctuation(self):
        # Pauses are derived from punctuation downstream, so it must survive.
        out = _clean_text("In 1944, he left.")
        assert out.endswith(".")


class TestAbbreviations:
    def test_initialisms_are_spelled_letter_by_letter(self):
        assert _clean_text("the USA and the FBI") == "the U S A and the F B I"

    def test_speed_units_expand(self):
        out = _clean_text("It moved at 300 kph.")
        assert "kilometers per hour" in out


class TestPreservedContent:
    def test_parentheses_are_preserved(self):
        raw = "He waited (for hours) in the dark."
        assert _clean_text(raw) == raw

    def test_square_brackets_are_preserved(self):
        raw = "The tank rolled in. [pause] It stopped."
        assert _clean_text(raw) == raw

    @pytest.mark.parametrize("ch", [".", "?", "!", ","])
    def test_sentence_punctuation_survives(self, ch):
        assert ch in _clean_text(f"Wait{ch} what is it")


class TestIdempotence:
    """Re-cleaning already-normalized text must be a no-op.

    The TTS path cleans text, and the voice-test harness re-cleans cached
    lines, so running it twice has to be safe.
    """

    def test_clean_text_is_idempotent(self):
        raw = "In 1944, the USA spent $5 million -- 1,200 troops, 300 kph."
        once = _clean_text(raw)
        assert _clean_text(once) == once

    def test_idempotent_for_sentences(self):
        raw = "Wait... what? Now! Yes, go."
        once = _clean_text(raw)
        assert _clean_text(once) == once
