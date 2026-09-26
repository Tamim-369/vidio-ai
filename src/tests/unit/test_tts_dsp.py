"""Characterization tests for the post-synthesis DSP helpers.

``tts_dsp`` is the shared post-synthesis seam. It is the highest-risk area to
move during a restructure because it is pure numpy/SoX and has no I/O, so its
behaviour is fully observable here.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agents.voiceover import dsp as tts_dsp
from src.agents.voiceover import dsp, pauses, timing

SR = 24000


def _tone(seconds: float = 1.0, freq: float = 220.0, amp: float = 0.5) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestPunctuationDetection:
    @pytest.mark.parametrize("token,expected", [
        ("word.", "."),
        ("word!", "!"),
        ("word?", "?"),
        ("word,", ","),
        ("word;", ";"),
        ("word:", ":"),
        ("word-", "-"),
        ("word\u2013", "\u2013"),   # en dash
        ("word\u2014", "\u2014"),   # em dash
    ])
    def test_trailing_punctuation_is_found(self, token, expected):
        assert pauses._trailing_punct(token) == expected

    @pytest.mark.parametrize("token", ['"Total."', "'war,'", "\u201dDone!\u201d"])
    def test_trailing_quotes_are_stripped_before_detection(self, token):
        assert pauses._trailing_punct(token) in (".", ",", "!")

    @pytest.mark.parametrize("token", ["word", "", "12.5", "3,000"])
    def test_no_punctuation_returns_none(self, token):
        # Decimal/range numbers must not be mistaken for sentence punctuation.
        assert pauses._trailing_punct(token) is None


class TestSilenceGapDetection:
    def test_detects_a_trailing_silence(self):
        audio = np.concatenate([_tone(0.5), np.zeros(int(0.3 * SR), dtype=np.float32)])
        gaps = tts_dsp._detect_silence_gaps(audio, SR)
        assert gaps, "expected at least one silence gap"
        assert any(d >= 0.2 for _, d in gaps)

    def test_short_audio_returns_empty(self):
        assert tts_dsp._detect_silence_gaps(_tone(0.05), SR) == []


class TestNoiseGate:
    def test_quiet_tail_is_gated_down(self):
        audio = np.concatenate([_tone(0.5), np.full(int(0.5 * SR), 0.0005, dtype=np.float32)])
        gated = tts_dsp._noise_gate(audio)
        tail = gated[int(0.6 * SR):]
        assert np.max(np.abs(tail)) < np.max(np.abs(audio[int(0.6 * SR):]))

    def test_speech_level_is_preserved(self):
        audio = _tone(1.0)
        gated = tts_dsp._noise_gate(audio)
        # The gate must not swallow the body of the signal.
        assert np.max(np.abs(gated)) > 0.3 * np.max(np.abs(audio))

    def test_does_not_shorten_the_clip(self):
        audio = _tone(1.0)
        assert len(tts_dsp._noise_gate(audio)) == len(audio)


class TestDeShout:
    def test_all_caps_words_become_title_case(self):
        assert tts_dsp._de_shout("THIS IS FINE") == "This Is Fine"

    def test_trailing_punctuation_is_preserved(self):
        assert tts_dsp._de_shout("THIS IS FINE!") == "This Is Fine!"

    def test_repeated_marks_collapse_when_caps_present(self):
        assert tts_dsp._de_shout("STOP!!") == "Stop!"

    @pytest.mark.parametrize("text", ["", "already normal text", None])
    def test_text_without_uppercase_returns_unchanged(self, text):
        # The guard short-circuits when there is no uppercase to neutralise,
        # so lowercase input (including "really!!") is returned verbatim.
        assert tts_dsp._de_shout(text) == text


class TestTimeStretch:
    def test_factor_of_one_is_a_no_op_lengthwise(self):
        audio = _tone(1.0)
        out = tts_dsp._time_stretch(audio, SR, 1.0)
        assert abs(len(out) - len(audio)) < SR * 0.05

    def test_nonpositive_factor_returns_input(self):
        audio = _tone(0.5)
        assert tts_dsp._time_stretch(audio, SR, 0) is audio
        assert tts_dsp._time_stretch(audio, SR, -1.0) is audio

    def test_slowdown_is_clamped_to_the_safe_floor(self):
        # A factor below 0.97 reads as a sudden speed drop and is clamped.
        audio = _tone(2.0)
        out = tts_dsp._time_stretch(audio, SR, 0.5)
        # Clamped to 0.97 -> output must be no longer than a 3% stretch.
        assert len(out) <= len(audio) * 1.06


class TestPacing:
    def test_wps_band_settings_are_sane(self):
        from src.agents.voiceover.dsp import TTS_MAX_WPS, TTS_MIN_WPS
        assert 0 < TTS_MIN_WPS < TTS_MAX_WPS

    def test_too_fast_line_is_slowed(self):
        # 30 words crammed into 1s of speech is far above the band ceiling.
        text = " ".join(["word"] * 30)
        audio = _tone(1.0)
        out = tts_dsp._normalize_pacing(audio, SR, text, {})
        assert len(out) >= len(audio) * 0.95

    def test_short_text_is_untouched(self):
        audio = _tone(0.4)
        assert tts_dsp._normalize_pacing(audio, SR, "hi", {}) is audio


class TestPostprocessLine:
    def test_returns_final_and_raw(self):
        audio = _tone(1.0)
        text = "The tank rolled in. It stopped."
        final, raw = tts_dsp.postprocess_line(audio, SR, "chatterbox", text, {})
        assert final is not None and raw is not None
        assert len(raw) == len(audio)
        # Finalization appends a tail pad and a lead-in, so it is longer.
        assert len(final) > len(raw)

    def test_is_deterministic(self):
        audio = _tone(1.0)
        text = "The tank rolled in. It stopped."
        f1, _ = tts_dsp.postprocess_line(audio.copy(), SR, "chatterbox", text, {})
        f2, _ = tts_dsp.postprocess_line(audio.copy(), SR, "chatterbox", text, {})
        assert np.array_equal(f1, f2)

    def test_output_is_clipped(self):
        audio = _tone(1.0, amp=0.99)
        final, _ = tts_dsp.postprocess_line(audio, SR, "chatterbox", "Hello there.", {})
        assert np.max(np.abs(final)) <= 0.9501


# --- narration rate ----------------------------------------------------------

class TestApplyRate:
    """TTS_RATE stretches the synthesised line; pitch must survive it."""

    def _tone(self, seconds=1.0, freq=220.0, sr=24000):
        t = np.arange(int(seconds * sr)) / sr
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr

    def test_default_rate_is_a_slight_slowdown(self):
        from src.agents.voiceover.engine import TTS_RATE

        assert 0.85 < TTS_RATE < 1.0, TTS_RATE

    def test_slower_rate_lengthens_the_audio(self, monkeypatch):
        from src.agents.voiceover import engine as tts_mod

        samples, sr = self._tone(1.0)
        monkeypatch.setattr(tts_mod, "TTS_RATE", 0.5)
        out = tts_mod._apply_rate(samples, sr)
        assert len(out) == pytest.approx(len(samples) * 2, rel=0.06)

    def test_rate_of_one_is_a_passthrough(self, monkeypatch):
        from src.agents.voiceover import engine as tts_mod

        samples, sr = self._tone(0.2)
        monkeypatch.setattr(tts_mod, "TTS_RATE", 1.0)
        assert tts_mod._apply_rate(samples, sr) is samples

    def test_pitch_is_preserved(self, monkeypatch):
        """atempo is a time-stretch, not a resample, so 220Hz stays 220Hz."""
        from src.agents.voiceover import engine as tts_mod

        samples, sr = self._tone(1.0, 220.0)
        monkeypatch.setattr(tts_mod, "TTS_RATE", 0.5)
        out = tts_mod._apply_rate(samples, sr)
        out = out[len(out) // 4: len(out) // 2]
        spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
        peak_hz = np.fft.rfftfreq(len(out), 1 / sr)[int(np.argmax(spectrum))]
        assert abs(peak_hz - 220.0) < 8.0, f"pitch drifted to {peak_hz:.1f}Hz"

    def test_extreme_rates_are_chained_not_rejected(self, monkeypatch):
        """atempo only accepts 0.5-2.0, so slower rates chain factors."""
        from src.agents.voiceover import engine as tts_mod

        samples, sr = self._tone(0.4)
        monkeypatch.setattr(tts_mod, "TTS_RATE", 0.25)
        out = tts_mod._apply_rate(samples, sr)
        assert len(out) == pytest.approx(len(samples) * 4, rel=0.15)


class TestPauseAlignment:
    """_enforce_pauses() must actually use waveform word alignment.

    The aligner is imported inside the function and guarded by
    `except Exception: times = None`, so a stale import path fails silently and
    every line quietly falls back to char-proportional timing. That is exactly
    what a module move once caused, so the path is asserted directly.
    """

    def test_aligner_is_imported_and_called(self, monkeypatch):
        from src.agents.voiceover import dsp

        called = []

        def spy(audio, sr, text):
            called.append(text)
            return [(0.0, 0.4), (0.4, 0.8)]

        monkeypatch.setattr(timing, "word_times_from_waveform", spy)
        samples = _tone(0.9)
        sr = SR
        dsp._enforce_pauses(samples, sr, "one. two.")

        assert called, ("word alignment was never used; the char-proportional "
                        "fallback is silently taking over")

    def test_pause_lands_after_the_sentence_not_mid_word(self, monkeypatch):
        from src.agents.voiceover import dsp

        # Two bursts with a real silence between them: the "." sits on that gap.
        sr = 16000

        def tone(dur, freq):
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

        audio = np.concatenate([
            tone(0.4, 180), np.zeros(int(sr * 0.25), dtype=np.float32), tone(0.4, 180),
        ])
        out = dsp._enforce_pauses(audio, sr, "one. two.")
        # A 0.20s sentence pause on a 0.25s existing gap: the gap is trimmed
        # toward target, so the result is shorter than the input, not longer.
        assert len(out) < len(audio), "existing silence should be trimmed to target"
