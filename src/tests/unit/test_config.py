"""The voice registry's contract, and the couple of assets every render needs.

A voice is a dict of strings, and the pipeline reads it with .get() in a dozen
places, so a missing key does not raise at registration -- it raises later, on a
live render, after a quote has been generated and TTS has run. That is the most
expensive place a config mistake can surface, so the fields are checked here
against the real files on disk.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.voice_cast import voices, writing_styles

# The three clone voices this project is built around. Arnold is registered
# even when disabled, so a bad entry is caught before it is enabled.
REQUIRED_VOICES = ("donald-trump", "arnold-schwarzenegger", "andrew-tate")

ENABLED_VOICES = ("donald-trump", "arnold-schwarzenegger", "andrew-tate")


class TestVoiceRegistry:
    @pytest.mark.parametrize("voice_id", REQUIRED_VOICES)
    def test_voice_is_complete(self, voice_id, project_root):
        """Every field a render reads must be present and point at a real file.

        Checked as one test per voice rather than one per field: the fields are
        only meaningful together, and a voice that is half-configured is the
        only failure this guards.
        """
        v = voices.get_voice(voice_id)
        assert v is not None, f"{voice_id} is not registered"

        assert v.get("name"), "the pipeline logs VOICES[id]['name']"
        short = v.get("short_name")
        assert short and " " not in short, "a title cannot fit a two-word name"
        assert v.get("quote_authors"), "no byline to credit"

        assert v["engine"] == "chatterbox", "only chatterbox voices are supported"

        style_id = v.get("writing_style")
        assert style_id in writing_styles.WRITING_STYLES, f"unknown style {style_id!r}"

        for field in ("ref_audio", "face"):
            rel = v.get(field)
            assert rel, f"{voice_id} has no {field}"
            assert (project_root / rel).is_file(), f"{voice_id} {field} missing: {rel}"

    def test_the_enabled_set_is_exactly_the_expected_voices(self):
        # Pins both the members and the order, so a voice silently dropped from
        # rotation is noticed.
        assert [vid for vid, _ in voices.get_enabled_voices()] == list(ENABLED_VOICES)

    def test_unknown_voice_returns_none(self):
        assert voices.get_voice("does-not-exist") is None


class TestWritingStyles:
    @pytest.mark.parametrize("given", ["nope", None])
    def test_an_unknown_style_falls_back_to_narrator(self, given):
        assert writing_styles.get_style(given) is writing_styles.WRITING_STYLES["narrator"]


def test_music_source_is_present():
    """oogway.mp3 is the background bed; losing it breaks every render."""
    assert (Path(__file__).resolve().parents[2] / "music" / "oogway.mp3").is_file()
