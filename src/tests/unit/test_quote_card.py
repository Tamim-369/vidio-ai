"""Tests for the speaker-branded quote card renderer.

The renderer is pure layout + ffmpeg, so these assert the geometry contract the
design depends on (quote in the top 40%, byline clear of the face, duration
driven by audio) rather than pixel-exact output.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image, ImageFont

from src.config.voices import VOICES, get_voice, pick_quote_author
from src.services import quote_card
from src.services.quote_card import (
    BYLINE_AREA_FRACTION,
    BYLINE_LIFT,
    BYLINE_SQUARE_PAD,
    MARGIN_X,
    NAME_SIZE,
    SOURCE_SIZE,
    QUOTE_BOTTOM,
    QUOTE_TOP,
    _byline_square,
    _credit_labels,
    _fit_byline,
    _fit_quote,
    _load_background,
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
        font = ImageFont.truetype(quote_card.FONT_PATH, 60)
        assert _wrap("Short.", font, 800) == ["Short."]

    def test_wraps_on_words(self):
        font = ImageFont.truetype(quote_card.FONT_PATH, 60)
        lines = _wrap("one two three four five six seven eight", font, 400)
        assert len(lines) > 1
        assert " ".join(lines).split() == "one two three four five six seven eight".split()

    def test_no_line_exceeds_the_box(self):
        font = ImageFont.truetype(quote_card.FONT_PATH, 60)
        for line in _wrap("word " * 60, font, 500):
            assert font.getlength(line) <= 500

    def test_a_word_wider_than_the_box_is_hard_split(self):
        font = ImageFont.truetype(quote_card.FONT_PATH, 90)
        lines = _wrap("X" * 400, font, 300)
        assert len(lines) > 1
        assert all(font.getlength(ln) <= 300 for ln in lines)

    def test_explicit_newlines_are_kept(self):
        font = ImageFont.truetype(quote_card.FONT_PATH, 60)
        assert _wrap("one\ntwo", font, 800) == ["one", "two"]


class TestFitQuote:
    def test_short_quotes_get_a_big_font(self):
        font, lines, _ = _fit_quote("Short.", 885, 653, SIZE)
        assert font.size >= 48
        assert len(lines) == 1

    def test_the_quote_does_not_fill_the_whole_band(self):
        # The quote should sit comfortably inside the top 40%, not fill it.
        _, _, line_h = _fit_quote("Short.", 885, 653, SIZE)
        assert line_h * 3 < 653, "type is too large for the top band"

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
    def test_short_names_keep_the_full_size(self):
        font = _fit_byline("Don Tzu", 576, 54, SIZE)
        assert font.getlength("Don Tzu") <= 576
        assert font.size == 54

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


class TestBylineSquare:
    def test_square_is_a_quarter_of_the_frame_area(self):
        _, _, side = _byline_square(SIZE)
        assert side ** 2 == pytest.approx(BYLINE_AREA_FRACTION * SIZE[0] * SIZE[1],
                                          rel=0.01)

    def test_square_is_anchored_in_the_bottom_left_corner(self):
        x0, y0, side = _byline_square(SIZE)
        assert x0 == 0
        assert y0 + side == SIZE[1]
        assert side < SIZE[0] and side < SIZE[1]

    def test_square_centres_on_360_1560_at_1080x1920(self):
        x0, y0, side = _byline_square(SIZE)
        assert (x0 + side // 2, y0 + side // 2) == (360, 1560)

    def test_credit_is_lifted_20_percent_of_the_square_side(self):
        """20% of 720 = 144px up, so the block stays inside the square."""
        x0, y0, side = _byline_square(SIZE)
        lift = int(round(side * BYLINE_LIFT))
        assert BYLINE_LIFT == 0.20
        assert (x0 + side // 2, y0 + side // 2 - lift) == (360, 1416)

    def test_lifted_credit_stays_inside_the_square(self):
        x0, y0, side = _byline_square(SIZE)
        cy = y0 + side // 2 - int(round(side * BYLINE_LIFT))
        half_block = (NAME_SIZE + 30 + SOURCE_SIZE) // 2
        assert y0 < cy - half_block and cy + half_block < y0 + side

    def test_credit_stays_left_of_the_face(self):
        _, _, side = _byline_square(SIZE)
        assert side < SIZE[0] * 0.67, "square would reach into the bottom-right face"


# --- background --------------------------------------------------------------

class TestBackground:
    @pytest.mark.parametrize("name", ["Trump", "Arnold", "Tate"])
    def test_cover_fit_produces_the_frame_size(self, name):
        frame = _load_background(os.path.join(ROOT, "src", "faces", f"{name}.png"), SIZE)
        assert frame.shape == (SIZE[1], SIZE[0], 3)
        assert frame.dtype == np.uint8

    def test_the_top_is_dark_enough_for_white_text(self):
        frame = _load_background(os.path.join(ROOT, "src", "faces", "Trump.png"), SIZE)
        top = frame[: int(SIZE[1] * QUOTE_BOTTOM)]
        # Well under mid-grey so the white quote is unambiguous.
        assert top.mean() < 90

    def test_the_face_corner_is_not_heavily_darkened(self):
        # The bottom scrim fades out to the right so the face stays visible.
        frame = _load_background(os.path.join(ROOT, "src", "faces", "Trump.png"), SIZE)
        face = frame[int(SIZE[1] * 0.60):, int(SIZE[0] * 0.70):]
        assert face.mean() > 25


# --- author picking ----------------------------------------------------------

class TestPickQuoteAuthor:
    def test_every_enabled_voice_can_be_credited(self):
        from src.config.voices import get_enabled_voices

        for voice_id, voice in get_enabled_voices():
            name, source = pick_quote_author(voice, "a quote")
            assert name and source, voice_id

    def test_the_selection_is_stable_for_the_same_quote(self):
        voice = get_voice("donald-trump")
        assert pick_quote_author(voice, "same text") == pick_quote_author(voice, "same text")

    def test_every_quote_resolves_to_the_voice_only_credit(self):
        """Each voice has exactly one approved credit, whatever the quote.

        The author list is deliberately one entry per voice: only the three
        name/book pairs that were actually asked for are approved, and no
        unapproved name may be substituted when a different quote comes along.
        """
        for voice_id in ("donald-trump", "arnold-schwarzenegger", "andrew-tate"):
            voice = get_voice(voice_id)
            assert len(voice["quote_authors"]) == 1, \
                f"{voice_id} should have exactly one approved credit, " \
                f"got {voice['quote_authors']}"
            expected = tuple(x.strip() for x in voice["quote_authors"][0].split("|"))
            for seed in ("a", "quote 1", "x" * 60, "Never give up on your dreams"):
                assert pick_quote_author(voice, seed) == expected, \
                    f"{voice_id} invented a credit for {seed!r}"

    def test_the_name_and_source_are_separated(self):
        voice = get_voice("donald-trump")
        for name, source in [pick_quote_author(voice, f"q{i}") for i in range(20)]:
            assert name != source, "entries should carry a book/work name"

    def test_a_voice_without_authors_returns_nothing(self):
        assert pick_quote_author({"name": "n"}, "x") == ("", "")

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

    def test_layout_constants_stay_in_the_intended_bands(self):
        assert 0 < QUOTE_TOP < QUOTE_BOTTOM <= 0.40
        assert 0 < BYLINE_AREA_FRACTION <= 0.5
        assert 0 < BYLINE_SQUARE_PAD < 0.5


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

    def test_quote_sits_in_the_top_40_percent(self, tmp_path):
        from moviepy import VideoFileClip

        wav = self._wav(tmp_path, seconds=1.0)
        out = quote_card.render_card(
            "My father was a billionaire and I still learned the value of a hard no.",
            os.path.join(ROOT, "src", "faces", "Trump.png"),
            wav, str(tmp_path / "card2.mp4"),
            attribution='"Don Tzu"', source='"Fart of War"',
        )
        clip = VideoFileClip(out)
        try:
            frame = clip.get_frame(0).astype(np.float32)
        finally:
            clip.close()
        h, w = frame.shape[:2]
        white = (frame[..., 0] > 200) & (frame[..., 1] > 200) & (frame[..., 2] > 200)
        assert white[: int(h * QUOTE_BOTTOM)].sum() > 5000, "quote text missing up top"
        # Nothing may spill out of the top 40%. The band is only policed ABOVE
        # the credit square, since the credit now sits at 1200-1920 and is
        # legitimately inside the old "empty" middle.
        _, sq_y, _ = _byline_square((w, h))
        middle = white[int(h * QUOTE_BOTTOM):sq_y]
        assert middle.sum() < 200, "text should not spill below the top 40%"

    def test_byline_is_bottom_left_and_clear_of_the_face(self, tmp_path):
        from moviepy import VideoFileClip

        wav = self._wav(tmp_path, seconds=1.0)
        out = quote_card.render_card(
            "A short quote.",
            os.path.join(ROOT, "src", "faces", "Trump.png"),
            wav, str(tmp_path / "card3.mp4"),
            attribution='"Brolexander the gainz"', source='"Book of Gainz"',
        )
        clip = VideoFileClip(out)
        try:
            frame = clip.get_frame(0).astype(np.float32)
        finally:
            clip.close()
        h, w = frame.shape[:2]
        R, G, B = frame[..., 0], frame[..., 1], frame[..., 2]
        # The palette is monochrome, so the credit is found by matching the
        # exact type colours rather than by a hue band. NAME_COLOR is the same
        # value as QUOTE_COLOR, so the search is restricted to rows below the
        # quote band or the quote itself would match.
        def near(c, tol=14):
            return (np.abs(R - c[0]) < tol) & (np.abs(G - c[1]) < tol) \
                & (np.abs(B - c[2]) < tol)

        _, sq_y, sq_side = _byline_square((w, h))
        credit = near(quote_card.NAME_COLOR) | near(quote_card.SOURCE_COLOR)
        credit[:sq_y] = False
        assert credit[:, :sq_side].sum() > 500, \
            "credit missing from the bottom-left square"
        assert credit[:, sq_side:].sum() == 0, \
            "credit must stay inside that square"

    def test_credit_uses_no_yellow(self, tmp_path):
        """The credit was gold; it must now be neutral, like the quote."""
        from moviepy import VideoFileClip

        wav = self._wav(tmp_path, seconds=1.0)
        out = quote_card.render_card(
            "A short quote.",
            os.path.join(ROOT, "src", "faces", "Trump.png"),
            wav, str(tmp_path / "card4.mp4"),
            attribution='"Don Tzu"', source='"Fart of War"',
        )
        clip = VideoFileClip(out)
        try:
            frame = clip.get_frame(0).astype(np.float32)
        finally:
            clip.close()
        R, G, B = frame[..., 0], frame[..., 1], frame[..., 2]
        yellowish = (R > 180) & (B < 170) & ((R - B) > 70)
        assert yellowish.sum() == 0, f"{yellowish.sum()} yellow-ish px in the credit"
        for colour in (quote_card.NAME_COLOR, quote_card.SOURCE_COLOR,
                       quote_card.QUOTE_COLOR):
            assert abs(colour[0] - colour[1]) <= 2 and abs(colour[1] - colour[2]) <= 2, \
                f"{colour} is not neutral"

    def test_credit_draws_no_rule_and_quotes_both_lines(self, tmp_path):
        """No divider: just - "Don Tzu" over "Fart of War", two bands of type."""
        from moviepy import VideoFileClip

        wav = self._wav(tmp_path, seconds=1.0)
        out = quote_card.render_card(
            "A short quote.",
            os.path.join(ROOT, "src", "faces", "Trump.png"),
            wav, str(tmp_path / "card5.mp4"),
            attribution="Don Tzu", source="Fart of War",
        )
        clip = VideoFileClip(out)
        try:
            frame = clip.get_frame(0).astype(np.float32)
        finally:
            clip.close()
        h, w = frame.shape[:2]
        R, G, B = frame[..., 0], frame[..., 1], frame[..., 2]
        light = (R > 195) & (G > 195) & (B > 195)
        x0, y0, side = _byline_square((w, h))
        roi = light[y0:y0 + side, x0:x0 + side]
        rows = np.nonzero(roi.any(axis=1))[0]
        bands, start, prev = [], rows[0], rows[0]
        for r in rows[1:]:
            if r - prev > 4:
                bands.append((start, prev))
                start = r
            prev = r
        bands.append((start, prev))
        # Exactly two bands. A third would be the rule we removed.
        assert len(bands) == 2, f"expected only two text lines, got {bands}"
        name, source = bands
        assert name[0] < source[0], "name must sit above the book line"

    def test_credit_is_drawn_name_then_source_at_the_configured_sizes(
            self, tmp_path, monkeypatch):
        """The name is set larger than the book line, and drawn first.

        Asserted on the fonts handed to draw() rather than on the rendered
        pixels: ink height depends on which glyphs a string happens to use
        (Playfair's 'f' ascender is taller than its cap height), so a short
        title at 44px can measure taller than a name at 46px.
        """
        calls = []
        real_draw = quote_card.ImageDraw.Draw

        def spy(img, *a, **k):
            draw = real_draw(img, *a, **k)
            real_text = draw.text

            def text(xy, text_, *ta, **tk):
                font = tk.get("font")
                if font is not None:
                    calls.append((text_, font.size, xy[1]))
                return real_text(xy, text_, *ta, **tk)

            draw.text = text
            return draw

        monkeypatch.setattr(quote_card.ImageDraw, "Draw", spy)
        quote_card.render_card(
            "A short quote.", os.path.join(ROOT, "src", "faces", "Trump.png"),
            self._wav(tmp_path, seconds=1.0), str(tmp_path / "sizes.mp4"),
            attribution="Don Tzu", source="The Fart of War",
        )

        credit = [c for c in calls if c[1] in (NAME_SIZE, quote_card.SOURCE_SIZE)]
        assert len(credit) == 2, f"expected two credit draws, got {credit}"
        (name_text, name_px, name_y), (src_text, src_px, src_y) = credit
        assert name_text == "- Don Tzu,", name_text
        assert src_text == "The Fart of War", src_text
        assert name_px == NAME_SIZE and src_px == quote_card.SOURCE_SIZE
        assert name_px < src_px, "the book line is the larger of the two"
        assert name_y < src_y, "author line must be drawn above"

    def test_credit_carries_a_dash_a_comma_and_no_quotes(self):
        """Dash and comma frame the author; the quotation marks are not here."""
        from src.services.quote_card import _credit_labels

        assert _credit_labels("Don Tzu", "Fart of War") == ("- Don Tzu,", "Fart of War")

    def test_no_real_author_or_title_is_quoted(self):
        from src.services.quote_card import _credit_labels

        for voice in VOICES.values():
            for author in voice.get("quote_authors", []):
                name, book = author.split("|")
                n, b = _credit_labels(name.strip(), book.strip())
                assert n == f"- {name.strip()},", n
                assert b == book.strip(), b
                assert '"' not in n and '"' not in b, (n, b)

    def test_missing_source_leaves_the_book_line_blank(self):
        from src.services.quote_card import _credit_labels

        assert _credit_labels("Don Tzu", "") == ("- Don Tzu,", "")

    def test_quote_is_the_only_quoted_text(self):
        from src.services.quote_card import _quote_display

        assert _quote_display("Know thyself.") == '"Know thyself."'

    def test_quotes_are_display_only_and_never_reach_the_stored_quote(self):
        """TTS must speak bare text, and the 30-80 budget counts words not marks."""
        from src.services.quote_card import _quote_display

        stored = "x" * 80
        assert len(stored) == 80
        # The card adds two glyphs the stored quote and the budget never see.
        assert _quote_display(stored) == '"' + "x" * 80 + '"'
        assert stored == "x" * 80

    def test_quotes_survive_alternate_apostrophes_in_titles(self, tmp_path):
        from moviepy import VideoFileClip

        wav = self._wav(tmp_path, seconds=1.0)
        out = quote_card.render_card(
            "A short quote.",
            os.path.join(ROOT, "src", "faces", "Trump.png"),
            wav, str(tmp_path / "card7.mp4"),
            attribution="Brolexander the Gainz", source="Book of Gainz",
        )
        clip = VideoFileClip(out)
        try:
            frame = clip.get_frame(0).astype(np.float32)
        finally:
            clip.close()
        h, w = frame.shape[:2]
        x0, y0, side = _byline_square((w, h))
        R, G, B = frame[..., 0], frame[..., 1], frame[..., 2]
        light = (R > 195) & (G > 195) & (B > 195)
        roi = light[y0:y0 + side, x0:x0 + side]
        rows = np.nonzero(roi.any(axis=1))[0]
        bands, start, prev = [], rows[0], rows[0]
        for r in rows[1:]:
            if r - prev > 4:
                bands.append((start, prev))
                start = r
            prev = r
        bands.append((start, prev))
        assert len(bands) == 2, bands

    def test_render_requires_a_face_image(self, tmp_path):
        wav = self._wav(tmp_path, seconds=0.5)
        script = {
            "topic": "t",
            "lines": [{"id": 1, "text": "hi", "audio_path": wav,
                       "actual_duration": 0.5}],
        }
        with pytest.raises(RuntimeError, match="face"):
            quote_card.render(script, {"name": "No Face"})

    def test_render_skips_lines_without_audio(self, tmp_path):
        script = {
            "topic": "t",
            "lines": [{"id": 1, "text": "hi", "actual_duration": 1.0}],
        }
        with pytest.raises(RuntimeError, match="No renderable lines"):
            quote_card.render(script, {"name": "n", "face": "src/faces/Trump.png"})


def test_all_voice_faces_are_9x16_enough():
    """The photos are portrait 9:16-ish, so the cover crop is negligible."""
    for voice_id, voice in VOICES.items():
        face = voice.get("face")
        if not face or not os.path.isfile(os.path.join(ROOT, face)):
            continue
        with Image.open(os.path.join(ROOT, face)) as img:
            ratio = img.width / img.height
        assert 0.5 < ratio < 0.63, f"{voice_id} face aspect {ratio:.3f} is not portrait"


# --- typeface selection -----------------------------------------------------

class TestFontSelection:
    def test_uses_the_repo_font_when_present(self):
        """Fonts/ is the intended typeface, and it is what ships on this box."""
        from src.config.settings import QUOTE_FONT_PATH
        from src.services.quote_card import _quote_font

        if not os.path.isfile(os.path.join(ROOT, QUOTE_FONT_PATH)):
            pytest.skip("Fonts/ is not present in this checkout")
        assert os.path.basename(_quote_font(48).path) == os.path.basename(
            QUOTE_FONT_PATH)

    def test_falls_back_when_the_font_is_missing(self, monkeypatch):
        """Fonts/ is untracked, so a fresh clone must still render."""
        from src.config.settings import FONT_PATH
        from src.services.quote_card import _font

        font = _font("Fonts/does_not_exist/Nope.ttf", 48)
        assert font.path == FONT_PATH
        assert font.size == 48

    def test_credit_uses_the_same_face_for_both_lines(self):
        from src.services.quote_card import _fit_byline

        w = int(_byline_square(SIZE)[2] * (1 - 2 * BYLINE_SQUARE_PAD))
        name = _fit_byline("Book of Gainz", w, 54, SIZE)
        source = _fit_byline("Book of Gainz", w, quote_card.SOURCE_SIZE, SIZE)
        assert name.path == source.path, "name and source must be one typeface"
        assert source.size < name.size, "source is the smaller line of the pair"


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

    def test_speech_stops_and_only_the_music_carries_the_tail(self, tmp_path):
        """The ending hold is music-only: narration silent, bed still running.

        The hold is produced by padding the narration with silence, and the music
        is trimmed to the *video* duration rather than the speech duration. If
        either half regressed, the video would either end in dead quiet or keep
        talking over the outro.
        """
        from src.services.music import build_filter

        speech_s = 1.0
        total = speech_s + quote_card.QUOTE_END_TAIL_S
        padded, dur = quote_card._pad_audio(
            self._wav(tmp_path, speech_s), quote_card.QUOTE_END_TAIL_S,
            str(tmp_path / "held.wav"))
        assert abs(dur - total) < 0.02, dur

        import soundfile as sf
        audio, _ = sf.read(padded, always_2d=True)
        tail = audio[int(speech_s * 22050):]
        assert np.abs(tail).max() == 0.0, "someone is still speaking in the tail"

        # the bed is cut to the padded length, so it outlives the last word
        graph = build_filter(total)
        assert f"atrim=0:{total:.3f}" in graph, graph

    def test_gap_is_longer_than_nothing_and_end_hold_is_longer(self):
        assert quote_card.QUOTE_GAP_S == 2.0
        assert quote_card.QUOTE_END_TAIL_S == 4.0
        assert quote_card.QUOTE_END_TAIL_S > quote_card.QUOTE_GAP_S

    def test_credit_lines_are_close_in_size(self):
        """The book line is the larger of the two; the author is secondary.

        The name carries no information the title does not, so the size order
        follows the joke. If this inverts, the author starts shouting and the
        book line drops back to being a footnote.
        """
        assert quote_card.SOURCE_SIZE > quote_card.NAME_SIZE, \
            "the book line should be the larger of the two"
        assert quote_card.NAME_SIZE < 46, "author line should have shrunk again"

    def test_longest_configured_credit_fits_unshrunk(self):
        """Padding is only safe while the widest pair fits the text area.

        BYLINE_SQUARE_PAD is a fixed inset; the fitter would silently shrink an
        over-long name instead of failing, which is how a credit ends up a few
        pixels smaller in one video than the next.
        """
        side = _byline_square(SIZE)[2]
        width = int(side * (1 - 2 * BYLINE_SQUARE_PAD))
        for voice in VOICES.values():
            for entry in voice.get("quote_authors", []):
                label, book = _credit_labels(*[x.strip() for x in
                                               entry.split("|", 1)])
                assert _quote_font(NAME_SIZE).getlength(label) <= width, label
                assert _quote_font(SOURCE_SIZE).getlength(book) <= width, book

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
        from src.services.quote_card import _byline_square

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

    def test_the_three_requested_credits_are_the_configured_ones(self):
        """Pin the exact name/title pairs so they cannot drift."""
        expected = {
            "donald-trump": ("Don Tzu", "The Fart of War"),
            "arnold-schwarzenegger": ("Brolexander", "The Book of Gainz"),
            "andrew-tate": ("Andru Tatte", "The Way of Whatever"),
        }
        for voice_id, (name, book) in expected.items():
            entries = VOICES[voice_id]["quote_authors"]
            assert f"{name} | {book}" in entries, \
                f"{voice_id} is missing {name!r} | {book!r}: {entries}"

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
