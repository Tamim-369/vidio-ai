"""Typeface loading and text fitting for the quote cards.

Holds the font/size knobs and the measurement helpers (wrap, shrink-to-fit) so
card.py can stay about frame composition: where things sit and what gets drawn.
"""

import os
from functools import lru_cache

from PIL import ImageFont

# Playfair Display, from the repo's Fonts/ folder (SIL OFL 1.1). A serif suits
# the "wisdom" register far better than a UI sans. Fonts/ is not committed, so
# _font() falls back to FONT_PATH without it.
FONT_DIR = os.getenv("FONT_DIR", "Fonts/Playfair_Display/static")
QUOTE_FONT_PATH = os.getenv("QUOTE_FONT_PATH", f"{FONT_DIR}/PlayfairDisplay-Bold.ttf")
# The book/work line under the author reads better in the italic cut.
QUOTE_FONT_ITALIC_PATH = os.getenv("QUOTE_FONT_ITALIC_PATH", f"{FONT_DIR}/PlayfairDisplay-Italic.ttf")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# --- Type sizes (px at 1080x1920; scaled for other formats) ---
# Restrained on purpose: the quote should not fill the top band, so there is
# breathing room and the photo stays visible.
QUOTE_SIZE_MAX = 64
QUOTE_SIZE_MIN = 30

# The credit is placed by geometry, not a hand-tuned offset: the bottom-left
# quarter of the frame's AREA is treated as a square and the author block is
# centred in it. At 1080x1920 that is 720x720, centring the block well clear of
# the face, which sits bottom-right.
BYLINE_AREA_FRACTION = 0.25


def _scale(value: int, size: tuple) -> int:
    """Scale a 1080x1920-relative pixel value to the target frame size."""
    return int(round(value * size[1] / 1920))


@lru_cache(maxsize=64)
def _font(path: str, px: int) -> ImageFont.FreeTypeFont:
    """Load a typeface, falling back to the system font if the file is missing.

    Fonts/ ships outside git, so a fresh clone has no Playfair Display. Falling
    back keeps rendering working instead of failing every render on a fresh
    checkout; FONT_PATH is the DejaVu sans the rest of the project already uses.
    """
    try:
        return ImageFont.truetype(path, px)
    except (OSError, ValueError):
        return ImageFont.truetype(FONT_PATH, px)


def _quote_font(px: int) -> ImageFont.FreeTypeFont:
    return _font(QUOTE_FONT_PATH, px)


def _quote_font_italic(px: int) -> ImageFont.FreeTypeFont:
    return _font(QUOTE_FONT_ITALIC_PATH, px)



def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    """Greedy word wrap; over-long single words are hard-split to fit."""
    lines = []
    for para in str(text).split("\n"):
        words = para.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if font.getlength(candidate) <= max_width or not current:
                # A single word wider than the box must be broken up.
                if font.getlength(candidate) > max_width and not current:
                    chunk = ""
                    for ch in word:
                        if font.getlength(chunk + ch) > max_width and chunk:
                            lines.append(chunk)
                            chunk = ch
                        else:
                            chunk += ch
                    current = chunk
                else:
                    current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _fit_quote(text: str, max_width: int, max_height: int, size: tuple,
               line_spacing: float = 1.18) -> tuple:
    """Pick the largest font size whose wrapped quote fits the text box.

    Returns (font, lines, line_height). Shrinks from QUOTE_SIZE_MAX down to
    QUOTE_SIZE_MIN so short quotes render huge and long ones stay on screen.
    """
    max_px = _scale(QUOTE_SIZE_MAX, size)
    min_px = _scale(QUOTE_SIZE_MIN, size)
    start = int(round(max_px * (max_width / 1080.0)))
    floor_px = int(round(min_px * (max_width / 1080.0)))

    for px in range(start, max(floor_px, 1) - 1, -2):
        font = _quote_font(px)
        lines = _wrap(text, font, max_width)
        line_height = int(round(px * line_spacing))
        if len(lines) * line_height <= max_height and all(
            font.getlength(ln) <= max_width for ln in lines
        ):
            return font, lines, line_height

    # Nothing fit cleanly: return the smallest render and let it overflow the
    # box slightly rather than dropping text.
    font = _quote_font(max(floor_px, 1))
    return font, _wrap(text, font, max_width), int(round(max(floor_px, 1) * line_spacing))


def _byline_square(size: tuple) -> tuple:
    """The bottom-left "25% of the area" square: (x0, y0, side).

    A quarter of the frame's area, but forced to be a SQUARE, so its side is the
    square root of that area rather than a quarter of each dimension. At
    1080x1920 that is 720px anchored in the bottom-left corner.
    """
    w, h = size
    side = int(round((BYLINE_AREA_FRACTION * w * h) ** 0.5))
    return 0, h - side, side


def _fit_byline(text: str, max_width: int, size_px: int, size: tuple,
                italic: bool = False) -> ImageFont.FreeTypeFont:
    """Largest byline font at or below `size_px` that keeps `text` in `max_width`."""
    px = _scale(size_px, size)
    floor_px = max(int(round(px * 0.6)), 12)
    load = _quote_font_italic if italic else _quote_font
    for candidate in range(px, floor_px - 1, -1):
        font = load(candidate)
        if font.getlength(text) <= max_width:
            return font
    return load(floor_px)
