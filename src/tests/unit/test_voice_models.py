"""Tests for the Chatterbox watermarker being disabled.

Chatterbox constructs ``perth.PerthImplicitWatermarker()`` in ``__init__`` and
calls ``apply_watermark()`` on every ``generate()``. The watermark is designed to
be inaudible, so a missing swap produces audio that sounds correct, renders
correctly, and passes every smoke test while being traceable -- which is why it
needs an explicit assertion rather than an end-to-end check.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import src.agents.voiceover.models as models
from src.agents.voiceover.models import _NoWatermarker


class TestNoWatermarker:
    def test_returns_the_samples_untouched(self):
        wav = np.arange(8, dtype=np.float32)
        out = _NoWatermarker().apply_watermark(wav, sample_rate=24000)
        assert np.array_equal(out, wav)

    def test_returns_the_same_object_not_a_copy(self):
        # Passing the reference straight through is what makes it a true
        # no-op; a copy that happened to be equal would be hiding a transform.
        wav = np.zeros(4, dtype=np.float32)
        assert _NoWatermarker().apply_watermark(wav) is wav

    def test_does_not_modify_the_input(self):
        wav = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        before = wav.copy()
        _NoWatermarker().apply_watermark(wav, sample_rate=24000)
        assert np.array_equal(wav, before)

    def test_accepts_the_positional_and_keyword_call_shapes(self):
        wav = np.zeros(3, dtype=np.float32)
        wm = _NoWatermarker()
        assert np.array_equal(wm.apply_watermark(wav, 24000), wav)
        assert np.array_equal(wm.apply_watermark(wav, sample_rate=24000), wav)


class TestLoaderSwapsTheWatermarker:
    def _fake_chatterbox(self, monkeypatch):
        """Stand in for chatterbox so this test does not load real weights."""

        class FakeTTS:
            def __init__(self):
                # Mirrors the library: a watermarker is always present on a
                # freshly built model.
                self.watermarker = "perth.PerthImplicitWatermarker()"

            @classmethod
            def from_pretrained(cls, device):
                return cls()

        module = types.ModuleType("chatterbox")
        module.ChatterboxTTS = FakeTTS
        monkeypatch.setitem(sys.modules, "chatterbox", module)
        monkeypatch.setattr(models, "_chat_model", None)

    def test_the_loaded_model_no_longer_carries_perth(self, monkeypatch):
        self._fake_chatterbox(monkeypatch)
        model = models._get_chatterbox_model()
        assert isinstance(model.watermarker, _NoWatermarker)
        assert not isinstance(model.watermarker, str)

    def test_a_cached_model_keeps_the_disabled_watermarker(self, monkeypatch):
        self._fake_chatterbox(monkeypatch)
        models._get_chatterbox_model()
        # The second line reuses the cached model; the swap must survive.
        assert isinstance(models._get_chatterbox_model().watermarker, _NoWatermarker)


class TestAgainstTheInstalledLibrary:
    """Guard the assumption this fix rests on."""

    def test_chatterbox_still_exposes_a_watermarker_attribute(self):
        pytest.importorskip("chatterbox")
        from chatterbox import ChatterboxTTS
        assert hasattr(ChatterboxTTS, "from_pretrained")

    def test_chatterbox_still_calls_apply_watermark(self):
        # If a future chatterbox release drops or renames the call, this
        # stand-in silently becomes dead code and watermarking could return.
        pytest.importorskip("chatterbox")
        import inspect

        from chatterbox import ChatterboxTTS
        source = inspect.getsource(ChatterboxTTS)
        assert "apply_watermark(" in source
        assert "PerthImplicitWatermarker()" in source

    def test_our_stand_in_satisfies_the_call_signature(self):
        pytest.importorskip("chatterbox")
        import inspect

        from chatterbox import ChatterboxTTS
        source = inspect.getsource(ChatterboxTTS)
        assert "apply_watermark(wav" in source, "call shape changed; update _NoWatermarker"
        assert "sample_rate=self.sr" in source, "call shape changed; update _NoWatermarker"
        assert inspect.signature(_NoWatermarker.apply_watermark).parameters.keys() >= {"wav"}
