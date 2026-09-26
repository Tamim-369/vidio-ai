"""Speaker-branded quote cards.

Each line of a quote script becomes a single full-bleed card: the speaker's
photo is the background (the face already sits bottom-right in these images),
the quote is typeset large and centered in the TOP 40% of the frame, and the
fake author is credited bottom-left next to the face.

The video is exactly as long as the line's narration — there is no fixed target
duration, so a slow line yields a longer card and a fast line a shorter one.

No stock imagery, no crossfades, one still card per line.
"""

import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoClip, AudioFileClip

from src.config.settings import (
    FONT_PATH,
    QUOTE_END_TAIL_S,
    QUOTE_FONT_ITALIC_PATH,
    QUOTE_FONT_PATH,
    QUOTE_GAP_S,
    TEMP_DIR,
    VIDEO_FORMAT,
    VIDEO_RESOLUTIONS,
)
from src.config.voices import pick_quote_author
from src.services.video_assembler import concat_segments
from src.utils.file_helpers import output_path

FPS = 24

# --- Layout (fractions of frame height) ---
QUOTE_TOP = 0.06       # keeps the quote off the very top edge
QUOTE_BOTTOM = 0.40    # quote lives in the top 40%, above the face
MARGIN_X = 0.09        # left/right text inset
# The credit is placed by its own geometry rather than a hand-tuned offset: the
# bottom-left quarter of the frame's AREA is treated as a square, and the
# author block is centred in it. At 1080x1920 that square is 720x720 in the
# bottom-left corner, so the block centres on (360, 1560) — well clear of the
# face, which sits bottom-right.
BYLINE_AREA_FRACTION = 0.25
BYLINE_SQUARE_PAD = 0.10  # inset the text keeps from the square's own edges
# The credit sits above the square's centre so it reads as a caption hung under
# the quote rather than a label parked in the corner. Expressed as a fraction of
# the square's SIDE, so it scales with the frame and stays inside the square.
# 0.20 of the side = 144px at 1080x1920. For "20% of the frame height" instead
# (384px), set BYLINE_LIFT=0.53.
BYLINE_LIFT = 0.20
# --- Type sizes (px at 1080x1920; scaled for other formats) ---
# The quote is deliberately restrained: it should read without filling the top
# band, so there is breathing room around it and the photo stays visible.
QUOTE_SIZE_MAX = 64
QUOTE_SIZE_MIN = 30
# Both credit lines use the same typeface and weight, so the pair reads as one
# unit rather than a title and a footnote. The book/work line is deliberately
# the larger of the two: it carries the joke, and the author is only an
# attribution. Sized so the longest configured name and the longest title both
# fit the text area unshrunk, so no video ends up with a smaller credit.
NAME_SIZE = 40
SOURCE_SIZE = 44

# Quote type is knocked back 10% off pure white. Full-strength white on a photo
# is the first thing to look like a subtitle burn-in; a touch of headroom keeps
# it elegant and slightly less harsh against a bright portrait.
# The palette is deliberately monochrome — the credit used to be gold, which
# fought the portrait and read as a branding sticker. Hierarchy now comes from
# size and weight alone, not hue.
QUOTE_COLOR = (230, 230, 230)
NAME_COLOR = (230, 230, 230)
SOURCE_COLOR = (210, 210, 210)

# Scrim strength over the photo so white type stays readable on any image.
TOP_SCRIM = 0.72
BOTTOM_SCRIM = 0.66


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


def _load_background(image_path: str, size: tuple) -> np.ndarray:
    """Load a photo, cover-fit it to the frame, and darken it for text contrast."""
    w, h = size
    img = Image.open(image_path).convert("RGB")

    # Cover: scale so the frame is fully filled, then center-crop the overflow.
    if img.width / img.height > w / h:
        new_w, new_h = int(round(img.width * h / img.height)), h
    else:
        new_w, new_h = w, int(round(img.height * w / img.width))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    frame = np.asarray(img.crop((left, top, left + w, top + h)), dtype=np.float32)

    # Slight global knock-down, then a top gradient for the quote and a bottom
    # gradient for the byline. Smoothstep-ish easing keeps the bands invisible.
    frame *= 0.72
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]

    top_a = np.clip(1.0 - ys / (h * QUOTE_BOTTOM), 0.0, 1.0) ** 1.5 * TOP_SCRIM
    # The byline only needs contrast on the LEFT: fade the bottom scrim out
    # toward the right so the speaker's face stays at full brightness.
    bottom_band = np.clip((ys / h - 0.72) / (1.0 - 0.72), 0.0, 1.0) ** 1.5
    bottom_side = np.clip(1.0 - xs / (w * 0.62), 0.0, 1.0) ** 1.2
    bottom_a = bottom_band * bottom_side * BOTTOM_SCRIM
    alpha = np.clip(top_a + bottom_a, 0.0, 0.92)[:, :, None]

    return (frame * (1.0 - alpha)).astype(np.uint8)


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


def _pad_audio(audio_path: str, tail_s: float, out_path: str) -> tuple:
    """Write `audio_path` with `tail_s` of silence appended. Returns (path, dur).

    The card holds on screen through its own tail, so the beat is a silent hold
    rather than a frozen frame cutting to black.
    """
    import soundfile as sf

    data, sr = sf.read(audio_path, always_2d=True)
    tail = np.zeros((max(int(round(tail_s * sr)), 0), data.shape[1]), dtype=data.dtype)
    padded = np.concatenate([data, tail], axis=0)
    sf.write(out_path, padded, sr)
    return out_path, len(padded) / sr


def _credit_labels(attribution: str, source: str) -> tuple:
    """The two displayed credit lines.

        - Don Tzu,
        The Fart of War

    Unquoted on purpose: the quotation marks belong to the quote itself, not to
    the credit. The author keeps the leading dash and takes a trailing comma as
    the only ornament -- the comma is what makes the dash-and-comma read as one
    credit rather than a bullet stuck to a heading.
    Split out so the exact strings are assertable without rendering a frame.
    """
    return f"- {attribution},", (source or "")


def _quote_display(quote: str) -> str:
    """The quote as shown on the card, wrapped in quotation marks.

    Applied at RENDER time only, never to the stored quote. The TTS speaks the
    bare text (reading the marks aloud would be nonsense) and the 30-80
    character budget still measures the words, not the punctuation we add for
    looks. The card auto-fits, so an 80-character quote renders as 82 glyphs and
    simply shrinks by a hair if it no longer fits.
    """
    return f'"{quote}"'


def render_card(quote: str, image_path: str, audio_path: str, out_path: str,
                attribution: str = "", source: str = "",
                duration: float = None, size: tuple = None,
                tail_s: float = 0.0) -> str:
    """Render one quote card video of exactly `duration` seconds.

    `attribution`/`source` are the fake author credit (bottom-left). If omitted
    they are derived from the quote text via the voice's author list, which the
    caller normally supplies instead — this fallback keeps the function usable
    standalone. `tail_s` appends silence so the card holds past the last word
    (see QUOTE_GAP_S / QUOTE_END_TAIL_S).
    """
    size = size or VIDEO_RESOLUTIONS[VIDEO_FORMAT]
    w, h = size
    if duration is None:
        duration = AudioFileClip(audio_path).duration

    if tail_s > 0:
        os.makedirs(TEMP_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(out_path))[0]
        pad_path = os.path.join(TEMP_DIR, f"{stem}_pad.wav")
        audio_path, duration = _pad_audio(audio_path, tail_s, pad_path)
    else:
        pad_path = None

    frame = Image.fromarray(_load_background(image_path, size))
    draw = ImageDraw.Draw(frame)

    # --- Quote: centered in the top 40%, auto-wrapped and auto-fitted ---
    box_w = int(w * (1 - 2 * MARGIN_X))
    box_top = int(h * QUOTE_TOP)
    box_bottom = int(h * QUOTE_BOTTOM)
    font, lines, line_height = _fit_quote(_quote_display(quote), box_w,
                                          box_bottom - box_top, size)

    block_h = len(lines) * line_height
    y = box_top + ((box_bottom - box_top) - block_h) // 2
    for line in lines:
        draw.text((w // 2, y + line_height // 2), line, font=font,
                  fill=QUOTE_COLOR, anchor="mm")
        y += line_height

    # --- Credit: author block centred in the bottom-left 25%-area square ---
    if attribution:
        sq_x, sq_y, sq_side = _byline_square(size)
        cx = sq_x + sq_side // 2
        cy = sq_y + sq_side // 2 - int(round(sq_side * BYLINE_LIFT))
        text_max_w = int(sq_side * (1 - 2 * BYLINE_SQUARE_PAD))

        # The author carries a leading dash; the quotation marks live on the
        # quote above, not here. No rule and no other furniture -- the size step
        # is enough separation, and a drawn line just adds a hard edge to a
        # monochrome card.
        name_label, source_label = _credit_labels(attribution, source)
        name_font = _fit_byline(name_label, text_max_w, NAME_SIZE, size)
        name_h = name_font.size

        source_font = None
        if source:
            # Same face and weight as the name, just smaller: one unit.
            source_font = _fit_byline(source_label, text_max_w, SOURCE_SIZE, size)

        line_gap = _scale(30, size)
        block_h = name_h
        if source_font is not None:
            block_h += line_gap + source_font.size

        top = cy - block_h // 2
        draw.text((cx, top + name_h // 2), name_label, font=name_font,
                  fill=NAME_COLOR, anchor="mm")
        if source_font is not None:
            draw.text((cx, top + name_h + line_gap + source_font.size // 2),
                      source_label, font=source_font, fill=SOURCE_COLOR,
                      anchor="mm")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # One still card for the whole line: frames are the same array every tick.
    clip = VideoClip(lambda t: np.asarray(frame), duration=duration)
    clip = clip.with_audio(AudioFileClip(audio_path))
    clip.write_videofile(out_path, fps=FPS, codec="libx264", audio_codec="aac",
                         preset="veryfast", logger=None)
    clip.close()

    # The padded wav is scratch; temp/ is wiped between runs but do not rely
    # on that for a file this render created.
    if pad_path:
        try:
            os.remove(pad_path)
        except OSError:
            pass

    return out_path


def render(script: dict, voice: dict) -> str:
    """Render every line of a quote script as a card, then concatenate.

    Mirrors video_assembler.assemble(script) so the pipeline can swap one call
    for the other: one still card per line. Each card is its spoken audio plus
    a tail of silence — QUOTE_GAP_S after every quote but the last, and the
    longer QUOTE_END_TAIL_S on the final one, so the video never cuts off on the
    last word.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    image_path = voice.get("face")
    if not image_path or not os.path.exists(image_path):
        raise RuntimeError(
            f"Voice {voice.get('name')!r} has no usable face image "
            f"(got {image_path!r}); add a 'face' entry in src/config/voices.py"
        )

    renderable = [ln for ln in script["lines"] if ln.get("audio_path")]
    last_index = len(renderable) - 1

    segment_paths = []
    for position, line in enumerate(renderable):
        audio_path = line["audio_path"]
        tail_s = QUOTE_END_TAIL_S if position == last_index else QUOTE_GAP_S

        # A line may pin its own credit; otherwise the voice's author list picks
        # one deterministically from the quote text. Pinning matters when the
        # quote and the byline are chosen together (a hand-written script), where
        # a hash of the text could easily land on a different name.
        attribution, source = pick_quote_author(voice, seed=line["text"])
        if line.get("attribution"):
            attribution = line["attribution"]
        if line.get("source"):
            source = line["source"]
        seg = os.path.join(TEMP_DIR, f"card_{line['id']}.mp4")
        print(f"  [quote-card] Line {line['id']}: {line['actual_duration']:.1f}s "
              f"+{tail_s:.1f}s — {attribution or '(no byline)'}")
        render_card(
            quote=line["text"],
            image_path=image_path,
            audio_path=audio_path,
            out_path=seg,
            attribution=attribution,
            source=source,
            duration=line["actual_duration"],
            tail_s=tail_s,
        )
        segment_paths.append(seg)

    if not segment_paths:
        raise RuntimeError("No renderable lines in script")

    out = output_path(script["topic"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"  [quote-card] Concatenating {len(segment_paths)} cards → {out}")
    concat_segments(segment_paths, out)

    for p in segment_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    return out
