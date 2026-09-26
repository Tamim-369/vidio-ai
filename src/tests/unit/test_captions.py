"""Characterization tests for the word-timing model in src/agents/voiceover/timing.py.

The quote card shows a whole line at once, so there are no subtitles any more.
Word timing survives because tts_dsp._enforce_pauses() uses it to place pauses
at real sentence boundaries in the narration. The invariants that matter are:
times are monotonic, start at ~0, stay inside the clip, and one entry per word.

The algorithm splits bursts at real silences, so the tests use synthetic
waveforms rather than real TTS audio to keep them fast and offline.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agents.voiceover import timing as captions

SR = 24000


def _speech(seconds: float, seed: int = 0) -> np.ndarray:
    """Noisy band-limited signal that reads as 'speech energy'."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    return (0.4 * np.sin(2 * np.pi * 180 * t) + 0.2 * rng.standard_normal(n)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _assert_valid(times, n_words, duration):
    assert len(times) == n_words
    prev = -1.0
    for start, end in times:
        assert start >= 0, "no word may start before 0"
        assert end >= start, "end must not precede start"
        assert start >= prev - 1e-6, "timings must be monotonic"
        if duration:
            assert start <= duration + 0.5, "word starts inside the clip"
        prev = start


class TestWordTiming:
    def test_one_entry_per_word(self):
        text = "the tank rolled in and stopped"
        times = captions.word_times_from_waveform(_speech(3.0), SR, text)
        _assert_valid(times, len(text.split()), 3.0)

    def test_empty_text_yields_no_timings(self):
        # The one place an empty word list is handled: the model returns before
        # reaching the proportional fallback.
        assert captions.word_times_from_waveform(_speech(1.0), SR, "") == []
        assert captions.word_times_from_waveform(_speech(1.0), SR, None) == []

    def test_tiny_clip_falls_back_to_proportional(self):
        # Fewer than two RMS hops: not enough to find a burst boundary.
        tiny = _speech(0.0001)
        text = "one two three"
        times = captions.word_times_from_waveform(tiny, SR, text)
        _assert_valid(times, 3, 0.0)

    def test_pauses_separate_bursts(self):
        # A real silence mid-clip must split into two bursts, so the words
        # after the gap start after it rather than being spread across it.
        y = np.concatenate([_speech(1.0), _silence(0.4), _speech(1.0)])
        text = "first half here second half there"
        times = captions.word_times_from_waveform(y, SR, text)
        _assert_valid(times, len(text.split()), 2.4)
        gap_start = 1.0
        later = [s for s, _ in times if s > gap_start + 0.05]
        assert later, "some words should start after the silence"

    def test_stereo_is_downmixed(self):
        stereo = np.stack([_speech(1.0), _speech(1.0)], axis=1)
        text = "hello world"
        times = captions.word_times_from_waveform(stereo, SR, text)
        _assert_valid(times, 2, 1.0)

    def test_is_deterministic(self):
        y = _speech(2.0)
        text = "same input same output every time"
        assert captions.word_times_from_waveform(y, SR, text) == \
               captions.word_times_from_waveform(y, SR, text)

    def test_more_words_than_bursts_is_safe(self):
        y = _speech(0.5)
        times = captions.word_times_from_waveform(y, SR, " ".join(["w"] * 12))
        _assert_valid(times, 12, 0.5)

    def test_array_loaded_from_a_real_wav(self, tmp_path):
        """The actual call pattern: tts_dsp reads the wav, then times the array.

        captions used to own the file read (align_words); that moved into
        tts_dsp, so this pins the hand-off rather than the removed reader.
        """
        import soundfile as sf
        p = tmp_path / "a.wav"
        sf.write(str(p), _speech(1.5), SR)
        y, sr = sf.read(str(p), dtype="float32")
        times = captions.word_times_from_waveform(y, sr, "hello there world")
        _assert_valid(times, 3, 1.5)


class TestProportionalTiming:
    def test_distributes_across_duration(self):
        times = captions._proportional_timing(["a", "b", "c", "d"], 2.0)
        _assert_valid(times, 4, 2.0)

    def test_segments_are_equal_width(self):
        # The fallback spreads evenly; it does not weight by word length.
        times = captions._proportional_timing(["a", "extraordinarily", "b"], 3.0)
        widths = [round(e - s, 6) for s, e in times]
        assert widths == [1.0, 1.0, 1.0]

    def test_zero_duration_uses_default_pace(self):
        # A non-positive duration falls back to 0.35s per word.
        times = captions._proportional_timing(["a", "b"], 0.0)
        _assert_valid(times, 2, 0.0)
        assert round(times[-1][1], 6) == round(0.70, 6)

    def test_empty_word_list_raises(self):
        # PRE-EXISTING BUG, pinned on purpose: n = len(words) with no guard, so
        # an empty list divides by zero. If you want this fixed it belongs in
        # its own commit: return [] when there are no words, before dividing.
        with pytest.raises(ZeroDivisionError):
            captions._proportional_timing([], 2.0)

    def test_empty_word_list_with_zero_duration_raises(self):
        # The other route to the same bug: duration <= 0 rewrites duration as
        # len(words) * 0.35, which is 0.0 for an empty list, then 0 / 0.
        with pytest.raises(ZeroDivisionError):
            captions._proportional_timing([], 0.0)
