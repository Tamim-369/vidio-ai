"""Characterization tests for the voice registry and writing styles.

Pins the registry shape so a restructure cannot silently drop a voice, break a
ref-audio path, or change the enabled set.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.voice_cast import voices, writing_styles

# The three clone voices this project is built around.
REQUIRED_VOICES = ("donald-trump", "arnold-schwarzenegger", "andrew-tate")


class TestVoiceRegistry:
    @pytest.mark.parametrize("voice_id", REQUIRED_VOICES)
    def test_required_voice_exists(self, voice_id):
        assert voices.get_voice(voice_id) is not None

    @pytest.mark.parametrize("voice_id", REQUIRED_VOICES)
    def test_required_voice_is_enabled(self, voice_id):
        assert voices.get_voice(voice_id).get("enabled") is True

    @pytest.mark.parametrize("voice_id", REQUIRED_VOICES)
    def test_required_voice_uses_chatterbox(self, voice_id):
        assert voices.get_voice(voice_id)["engine"] == "chatterbox"

    @pytest.mark.parametrize("voice_id", REQUIRED_VOICES)
    def test_required_voice_reference_audio_exists(self, voice_id, project_root):
        # A missing ref file would fail deep inside TTS at render time.
        ref = voices.get_voice(voice_id)["ref_audio"]
        assert (project_root / ref).is_file(), f"missing ref audio for {voice_id}: {ref}"

    @pytest.mark.parametrize("voice_id", REQUIRED_VOICES)
    def test_required_voice_has_a_known_writing_style(self, voice_id):
        style_id = voices.get_voice(voice_id)["writing_style"]
        assert style_id in writing_styles.WRITING_STYLES

    def test_enabled_set_is_exactly_the_three_clones(self):
        # The legacy Pocket narrator stays registered but out of rotation.
        assert [vid for vid, _ in voices.get_enabled_voices()] == list(REQUIRED_VOICES)

    def test_unknown_voice_returns_none(self):
        assert voices.get_voice("does-not-exist") is None

    def test_get_all_includes_disabled(self):
        assert "narrator" in voices.get_all_voices()


class TestWritingStyles:
    def test_unknown_style_falls_back_to_narrator(self):
        assert writing_styles.get_style("nope") is writing_styles.WRITING_STYLES["narrator"]

    def test_none_falls_back_to_narrator(self):
        assert writing_styles.get_style(None) is writing_styles.WRITING_STYLES["narrator"]

    @pytest.mark.parametrize("style_id", ["narrator", "trump", "arnold"])
    def test_core_styles_have_a_persona(self, style_id):
        assert writing_styles.get_style(style_id)["persona"].strip()


class TestSettings:
    """Each value is asserted where it now lives, with its consumer."""

    def test_tts_pacing_band_is_ordered(self):
        from src.agents.voiceover.dsp import TTS_MAX_WPS, TTS_MIN_WPS
        assert 0 < TTS_MIN_WPS < TTS_MAX_WPS

    def test_video_resolutions_cover_the_configured_format(self):
        from src.agents.visuals.card import VIDEO_FORMAT, VIDEO_RESOLUTIONS
        assert VIDEO_FORMAT in VIDEO_RESOLUTIONS

    def test_output_and_temp_dirs_are_relative(self):
        from src.agents.video.artifacts import TEMP_DIR
        from src.agents.visuals.card import OUTPUT_DIR
        # Relative paths keep the project portable across machines.
        assert not OUTPUT_DIR.startswith("/")
        assert not TEMP_DIR.startswith("/")

    def test_music_source_is_present(self):
        # oogway.mp3 is the background bed; losing it breaks every render.
        assert (Path(__file__).resolve().parents[2] / "music" / "oogway.mp3").is_file()
