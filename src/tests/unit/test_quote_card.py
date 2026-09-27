"""Tests for the speaker-branded quote card renderer.

The renderer is pure layout + ffmpeg, so these assert the geometry contract the
design depends on (quote in the top 40%, byline clear of the face, duration
driven by audio) rather than pixel-exact output.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import ImageFont

from src.agents.voice_cast.voices import VOICES, get_voice, pick_quote_author
from src.agents.visuals import card as quote_card
from src.agents.visuals.card import (
    BYLINE_SQUARE_PAD,
    MARGIN_X,
    NAME_SIZE,
    QUOTE_BOTTOM,
    QUOTE_TOP,
    _credit_labels,
)
from src.agents.visuals.layout import (
    FONT_PATH,
    _byline_square,
    _fit_byline,
    _fit_quote,
    _quote_font,
    _wrap,
)

SIZE = (1080, 1920)
# src/tests/unit/test_quote_card.py -> repo root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


# --- text fitting ------------------------------------------------------------

class TestWrap:
    def test_short_text_is_one_line(self):
        font = ImageFont.truetype(FONT_PATH, 60)
        assert _wrap("Short.", font, 800) == ["Short."]

    def test_wraps_on_words(self):
        font = ImageFont.truetype(FONT_PATH, 60)
        lines = _wrap("one two three four five six seven eight", font, 400)
        assert len(lines) > 1
        assert " ".join(lines).split() == "one two three four five six seven eight".split()

    def test_no_line_exceeds_the_box(self):
        font = ImageFont.truetype(FONT_PATH, 60)
        for line in _wrap("word " * 60, font, 500):
            assert font.getlength(line) <= 500

    def test_a_word_wider_than_the_box_is_hard_split(self):
        font = ImageFont.truetype(FONT_PATH, 90)
        lines = _wrap("X" * 400, font, 300)
        assert len(lines) > 1
        assert all(font.getlength(ln) <= 300 for ln in lines)

    def test_explicit_newlines_are_kept(self):
        font = ImageFont.truetype(FONT_PATH, 60)
        assert _wrap("one\ntwo", font, 800) == ["one", "two"]


class TestFitQuote:
    def test_short_quotes_get_a_big_font(self):
        font, lines, _ = _fit_quote("Short.", 885, 653, SIZE)
        assert font.size >= 48
        assert len(lines) == 1

    def test_longer_quotes_shrink_to_fit(self):
        # The fixture has to be long enough to overflow the box at the starting
        # size for whichever typeface is active, or nothing shrinks and this
        # asserts nothing. 600 chars overflows Playfair Display (the repo font)
        # but not DejaVu, so keep it comfortably past that.
        short = _fit_quote("Short.", 885, 653, SIZE)[0].size
        long = _fit_quote("word " * 120, 885, 653, SIZE)[0].size
        assert long < short

    def test_wrapped_block_always_fits_the_box(self):
        box_w = int(SIZE[0] * (1 - 2 * MARGIN_X))
        box_h = int(SIZE[1] * QUOTE_BOTTOM) - int(SIZE[1] * QUOTE_TOP)
        for text in ("Short.", "word " * 60, "word " * 200, "A " * 400):
            font, lines, line_h = _fit_quote(text, box_w, box_h, SIZE)
            assert all(font.getlength(ln) <= box_w for ln in lines), text[:20]

    def test_font_never_goes_below_the_floor(self):
        # The floor scales with the text box width, so at 885px it is ~24px.
        font, _, _ = _fit_quote("word " * 2000, 885, 653, SIZE)
        assert font.size >= 24


# --- byline fitting ----------------------------------------------------------

class TestFitByline:
    def test_long_names_are_shrunk_to_fit_the_square(self):
        # The shrink path needs a name that genuinely overflows. No real author
        # name does any more (the longest, "Brolexander the Gainz", is 553px in
        # Playfair against a 576px limit), so use a synthetic one and measure it
        # in the ACTIVE face rather than the system fallback.
        width = int(_byline_square(SIZE)[2] * (1 - 2 * BYLINE_SQUARE_PAD))
        name = "Bartholomew Vanderstein III"
        assert _quote_font(54).getlength(name) > width, "fixture no longer overflows"
        font = _fit_byline(name, width, 54, SIZE)
        assert font.getlength(name) <= width
        assert font.size < 54

    def test_every_real_author_name_fits_at_a_readable_size(self):
        """No configured name may overflow the square or shrink to nothing.

        "Sarnold Achwarzenegger" is wider than the text area at the nominal
        54px, so the fitter drops it to ~50px. That is the intended behaviour
        and keeps the padding comfortable, but a name that had to shrink hard
        would mean the text area is too small, so bound it.
        """
        width = int(_byline_square(SIZE)[2] * (1 - 2 * BYLINE_SQUARE_PAD))
        for voice in VOICES.values():
            for author in voice.get("quote_authors", []):
                label, book = _credit_labels(*[x.strip() for x in
                                                author.split("|", 1)])
                font = _fit_byline(label, width, NAME_SIZE, SIZE)
                assert font.getlength(label) <= width, label
                assert font.size >= NAME_SIZE * 0.9, \
                    f"{label!r} shrank to {font.size}px"
                bfont = _fit_byline(book, width, quote_card.SOURCE_SIZE, SIZE)
                assert bfont.getlength(book) <= width, book


# --- background --------------------------------------------------------------

# --- author picking ----------------------------------------------------------

class TestPickQuoteAuthor:
    def test_the_selection_is_stable_for_the_same_quote(self):
        voice = get_voice("donald-trump")
        assert pick_quote_author(voice, "same text") == pick_quote_author(voice, "same text")

    def test_an_entry_without_a_separator_is_used_for_both(self):
        voice = {"quote_authors": ["Anon"]}
        assert pick_quote_author(voice, "x") == ("Anon", "Anon")


# --- config wiring -----------------------------------------------------------

class TestVoiceFaces:
    @pytest.mark.parametrize("voice_id,image", [
        ("donald-trump", "Trump.png"),
        ("arnold-schwarzenegger", "Arnold.png"),
        ("andrew-tate", "Tate.png"),
    ])
    def test_voice_points_at_its_photo(self, voice_id, image):
        voice = get_voice(voice_id)
        assert voice["face"].endswith(image)
        assert os.path.isfile(os.path.join(ROOT, voice["face"]))

# --- end-to-end render -------------------------------------------------------

class TestRenderCard:
    def _wav(self, tmp_path, seconds=1.5, rate=22050):
        import soundfile as sf

        path = tmp_path / "line.wav"
        t = np.linspace(0, seconds, int(seconds * rate), endpoint=False)
        sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate)
        return str(path)

    def test_card_duration_follows_the_audio(self, tmp_path):
        from moviepy import VideoFileClip

        wav = self._wav(tmp_path, seconds=2.0)
        out = quote_card.render_card(
            "A short quote.", os.path.join(ROOT, "src", "faces", "Trump.png"),
            wav, str(tmp_path / "card.mp4"),
            attribution='"Don Tzu"', source='"Fart of War"',
        )
        clip = VideoFileClip(out)
        try:
            assert abs(clip.duration - 2.0) < 0.35, clip.duration
            assert clip.size == [SIZE[0], SIZE[1]]
            assert clip.audio is not None
        finally:
            clip.close()

    def test_quotes_are_display_only_and_never_reach_the_stored_quote(self):
        """TTS must speak bare text, and the 30-80 budget counts words not marks."""
        from src.agents.visuals.card import _quote_display

        stored = "x" * 80
        assert len(stored) == 80
        # The card adds two glyphs the stored quote and the budget never see.
        assert _quote_display(stored) == '"' + "x" * 80 + '"'
        assert stored == "x" * 80

    def test_render_requires_a_face_image(self, tmp_path):
        wav = self._wav(tmp_path, seconds=0.5)
        script = {
            "topic": "t",
            "lines": [{"id": 1, "text": "hi", "audio_path": wav,
                       "actual_duration": 0.5}],
        }
        with pytest.raises(RuntimeError, match="face"):
            quote_card.render(script, {"name": "No Face"})

# --- typeface selection -----------------------------------------------------

class TestFontSelection:
    def test_falls_back_when_the_font_is_missing(self, monkeypatch):
        """Fonts/ is untracked, so a fresh clone must still render."""
        from src.agents.visuals.layout import FONT_PATH
        from src.agents.visuals.layout import _font

        font = _font("Fonts/does_not_exist/Nope.ttf", 48)
        assert font.path == FONT_PATH
        assert font.size == 48

# --- pacing: gap between quotes, hold at the end -----------------------------

class TestPacing:
    def _wav(self, tmp_path, seconds, rate=22050):
        import soundfile as sf

        path = tmp_path / f"line_{seconds}.wav"
        t = np.linspace(0, seconds, int(seconds * rate), endpoint=False)
        sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate)
        return str(path)

    def test_pad_audio_appends_silence(self, tmp_path):
        import soundfile as sf

        src = self._wav(tmp_path, 1.0)
        out = str(tmp_path / "padded.wav")
        _, dur = quote_card._pad_audio(src, 1.5, out)
        assert abs(dur - 2.5) < 0.02, dur
        padded, _ = sf.read(out, always_2d=True)
        original, _ = sf.read(src, always_2d=True)
        assert np.array_equal(padded[: len(original)], original), "audio was altered"
        assert np.abs(padded[len(original):]).max() == 0.0, "tail is not silent"

    def test_card_holds_through_its_tail(self, tmp_path):
        from moviepy import VideoFileClip

        wav = self._wav(tmp_path, 1.0)
        out = quote_card.render_card(
            "A short quote.", os.path.join(ROOT, "src", "faces", "Trump.png"),
            wav, str(tmp_path / "tail.mp4"),
            attribution='"Don Tzu"', source='"Fart of War"',
            tail_s=4.0,
        )
        clip = VideoFileClip(out)
        try:
            assert abs(clip.duration - 5.0) < 0.35, clip.duration
        finally:
            clip.close()

    def test_render_totals_include_gap_and_end_hold(self, tmp_path, monkeypatch):
        """3 cards of 1s => 1+1+1 speech, 1s after each of the first two, 2s at the end."""
        from moviepy import VideoFileClip

        tails = []
        real = quote_card.render_card

        def spy(**kw):
            tails.append(kw["tail_s"])
            return real(duration=1.0, **{
                k: v for k, v in kw.items()
                if k in ("quote", "image_path", "audio_path", "out_path",
                         "attribution", "source", "tail_s")})

        monkeypatch.setattr(quote_card, "TEMP_DIR", str(tmp_path))
        monkeypatch.setattr(quote_card, "output_path",
                            lambda *a, **k: str(tmp_path / "out.mp4"))
        monkeypatch.setattr(quote_card, "render_card", spy)

        script = {
            "topic": "t",
            "lines": [
                {"id": i, "text": f"quote {i}",
                 "audio_path": self._wav(tmp_path, 1.0), "actual_duration": 1.0}
                for i in (1, 2, 3)
            ],
        }
        quote_card.render(script, {"name": "n", "face": "src/faces/Trump.png"})

        assert tails == [quote_card.QUOTE_GAP_S, quote_card.QUOTE_GAP_S,
                         quote_card.QUOTE_END_TAIL_S]
        assert quote_card.QUOTE_GAP_S == 2.0
        assert quote_card.QUOTE_END_TAIL_S == 4.0

        clip = VideoFileClip(str(tmp_path / "out.mp4"))
        try:
            expected = 3 * 1.0 + 2 * 2.0 + 4.0
            assert abs(clip.duration - expected) < 0.6, clip.duration
        finally:
            clip.close()

    def test_single_quote_gets_the_long_hold_not_a_gap(self, tmp_path, monkeypatch):
        tails = []
        real = quote_card.render_card

        def spy(**kw):
            tails.append(kw["tail_s"])
            return real(duration=1.0, **{
                k: v for k, v in kw.items()
                if k in ("quote", "image_path", "audio_path", "out_path",
                         "attribution", "source", "tail_s")})

        monkeypatch.setattr(quote_card, "TEMP_DIR", str(tmp_path))
        monkeypatch.setattr(quote_card, "output_path",
                            lambda *a, **k: str(tmp_path / "one.mp4"))
        monkeypatch.setattr(quote_card, "render_card", spy)

        script = {"topic": "t", "lines": [
            {"id": 1, "text": "only", "audio_path": self._wav(tmp_path, 1.0),
             "actual_duration": 1.0}]}
        quote_card.render(script, {"name": "n", "face": "src/faces/Trump.png"})

        assert tails == [quote_card.QUOTE_END_TAIL_S], tails


class TestPinnedCredits:
    """A hand-written script must be able to pin the byline it chose.

    The author is normally derived from a hash of the quote text, which is right
    for generated quotes but wrong when the quote and the credit were picked
    together: the hash can land on a different name than the one intended.
    """

    def _wav(self, tmp_path, seconds=0.6, rate=22050):
        import soundfile as sf

        path = tmp_path / "line.wav"
        t = np.linspace(0, seconds, int(seconds * rate), endpoint=False)
        sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
                 rate)
        return str(path)

    def test_line_attribution_overrides_the_hash(self, tmp_path, monkeypatch):
        from src.agents.visuals.layout import _byline_square

        monkeypatch.setattr(quote_card, "TEMP_DIR", str(tmp_path))
        monkeypatch.setattr(quote_card, "output_path",
                            lambda *a, **k: str(tmp_path / "pinned.mp4"))
        script = {"topic": "t", "lines": [{
            "id": 1, "text": "Never give up on your dreams, keep sleeping.",
            "audio_path": self._wav(tmp_path), "actual_duration": 0.6,
            "attribution": "Andru Tatte", "source": "The Way of Whatever"}]}
        quote_card.render(script, {"name": "n", "face": "src/faces/Tate.png"})

        from moviepy import VideoFileClip
        clip = VideoFileClip(str(tmp_path / "pinned.mp4"))
        try:
            frame = clip.get_frame(0).astype(np.float32)
        finally:
            clip.close()
        h, w = frame.shape[:2]
        x0, y0, side = _byline_square((w, h))
        roi = frame[y0:y0 + side, x0:x0 + side]
        lit = (roi[..., 0] > 195) & (roi[..., 1] > 195) & (roi[..., 2] > 195)
        assert lit.any(), "pinned credit did not render"

    def test_pick_quote_author_never_invents_a_name(self):
        """Whatever the hash picks must come from the configured list."""
        for voice in VOICES.values():
            allowed = {a.split("|")[0].strip() for a in voice.get("quote_authors", [])}
            if not allowed:
                continue
            for seed in ("a", "hello world", "x" * 50, "Never give up."):
                name, source = pick_quote_author(voice, seed=seed)
                assert name in allowed, (name, allowed)
                assert f"{name} | {source}" in voice["quote_authors"]
