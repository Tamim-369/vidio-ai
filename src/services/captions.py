"""Styled captions (subtitles) burned into video frames.

Chunked "karaoke" captions synced to the ACTUAL audio:

- each sentence is split into 3-4 spoken PARTS (boundaries nudge to
  punctuation); only the part currently being spoken is on screen
- word reveal times are derived from the line's WAV waveform (RMS energy),
  so the highlighted word tracks real speech — long/loud words get more
  time, and during natural pauses the current word holds (no robot ticking)
- part show/hide timing is PURELY visual: it follows the audio timeline and
  never alters the voiceover itself
- accent color is derived from VIDEO_STYLE (attraction -> amber, etc.)
- no background pill -- legibility comes from a soft drop shadow + stroke
- caption size/position adapt to the frame resolution

The captions are applied per line AFTER the Ken Burns/crossfade frames have
been rendered, so each segment caption rides that line's own timeline and
stays crisp (not blended by image crossfades).
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.config.settings import FONT_PATH

# Vibe -> accent color. Matches VIDEO_STYLE in settings.py.
THEME_ACCENTS = {
    "attraction": "#FFC94D",   # amber / high-energy
    "educational": "#5AF0FF",  # cyan / clear
    "motivational": "#FF7A3D", # orange / punchy
    "ad": "#FF4FAE",           # magenta / product
    "storytelling": "#FFE27A", # warm / narrative
}

# Fallback accent when the style is unknown or accent override is set.
DEFAULT_ACCENT = "#FFC94D"

# Caption layout constants (relative to frame size).
MAX_WIDTH_FRAC = 0.86          # caption block max width as fraction of frame
MARGIN_BOTTOM_FRAC = 0.16      # distance from frame bottom
LINE_SPACING = 1.30            # line height multiplier
STROKE_WIDTH_FRAC = 0.0035     # black outline for legibility
SHADOW_OFFSET_FRAC = 0.002     # drop shadow offset (of height)

# Word-timing (audio-forced alignment) constants.
HOP_S = 0.01                   # RMS envelope hop in seconds
SILENCE_RATIO = 0.025          # threshold = peak * this (activity gate)
MIN_WORD_GAP = 0.045           # force a word to hold at least this long (s)

# Chunked-part caption constants: a sentence is shown as 3-4 parts, each part
# appearing ONLY while one of its words is being spoken.
MIN_PARTS = 3                  # short/medium sentences -> 3 parts
MAX_PARTS = 4                  # long sentences (>= LONG_WORD_N words) -> 4
LONG_WORD_N = 12
PART_FADE_S = 0.12             # fade-in per part (avoids hard pops)
_PUNCT_PART = set(".,;:!?—…")


def accent_for_style(style: str, override: str = "") -> str:
    """Resolve the caption accent color for a video style."""
    if override:
        return override
    return THEME_ACCENTS.get(style, DEFAULT_ACCENT)


def _font_for(width: int) -> ImageFont.FreeTypeFont:
    """Pick a bold font scaled to the frame width."""
    size = max(26, int(width / 1080.0 * 52))
    return ImageFont.truetype(FONT_PATH, size)


def _wrap_words(words: list, font: ImageFont.FreeTypeFont, max_w: int) -> list:
    """Greedy word-wrap into a list of line-lists, lexed to max_w."""
    lines = []
    cur, cur_w = [], 0.0
    space_w = font.getlength(" ")
    for w in words:
        w_w = font.getlength(w)
        add = cur_w + (space_w + w_w if cur else w_w)
        if cur and add > max_w:
            lines.append(cur)
            cur, cur_w = [w], w_w
        else:
            cur.append(w)
            cur_w = add
    if cur:
        lines.append(cur)
    return lines


def _hex(hex_color: str, alpha: int = 255) -> tuple:
    """hex -> RGBA tuple."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)


def align_words(audio_path: str, text: str) -> list:
    """Return per-word (start, end) seconds, derived from audio RMS energy.

    Two-stage model:
    1. Segment the waveform into speech BURSTS separated by real silences
       (pauses). Word boundaries snap to those silence edges, so the
       highlight HOLDS on natural pauses instead of robot-ticking.
    2. Within each burst (may contain several words spoken without a pause)
       distribute word starts by cumulative speech energy weighted by word
       size, so longer/louder words get more of the burst's time.

    Falls back to proportional timing when the audio can't be read.
    """
    import soundfile as sf

    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    except Exception:
        return _proportional_timing((text or "").split(), 0.0)
    return word_times_from_waveform(np.asarray(y, dtype=np.float32), sr, text)


def word_times_from_waveform(y: np.ndarray, sr: int, text: str) -> list:
    """Core word-timing model used by align_words(), run on in-memory audio.

    Splits the waveform into speech bursts at real silences, then distributes
    each word's start by cumulative speech energy weighted by word size.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    words = (text or "").split()
    if not words:
        return []
    hop = max(1, int(sr * HOP_S))
    n_frames = len(y) // hop
    if n_frames < 2:
        return _proportional_timing(words, len(y) / sr)
    frame = y[:n_frames * hop].reshape(n_frames, hop)
    env = np.sqrt(np.mean(frame ** 2, axis=1) + 1e-12)
    times = np.arange(n_frames) * HOP_S

    thr = env.max() * SILENCE_RATIO
    active = env > thr

    # Speech bursts: contiguous active runs; merge gaps shorter than MIN_PAUSE.
    MIN_PAUSE_HOPS = 7  # ~70ms -- shorter dips are just word-internal energy
    labels = np.zeros(len(active), dtype=int)
    label = 0
    last_end = -10 ** 9
    for i, a in enumerate(active):
        if not a:
            continue
        if i - last_end > MIN_PAUSE_HOPS:
            label += 1
        labels[i] = label
        last_end = i
    bursts = []
    if label > 0:
        for lab in range(1, label + 1):
            idx = np.where(labels == lab)[0]
            bursts.append((times[idx[0]], times[idx[-1]] + HOP_S))
    if not bursts:
        # Still too flat to split -- treat whole clip as one continuous burst.
        bursts = [(0.0, times[-1] + HOP_S)]

    dur = len(y) / sr
    n_words = len(words)

    # Word size weights (cumulative fraction boundaries).
    weights = np.asarray([len(w) + 1 for w in words], dtype=float)
    cw = np.concatenate([[0.0], np.cumsum(weights)])
    fracs = cw / cw[-1]  # bit before smoothing

    # Distribute words across bursts: each burst claims its share of total
    # burst time; nearest index quantized so a burst can own >=1 word.
    burst_dur = np.asarray([b - a for a, b in bursts])
    total = burst_dur.sum()
    shares = np.round(burst_dur / total * n_words).astype(int)
    shares = np.maximum(shares, 1)
    while shares.sum() > n_words:
        shares[np.argmax(shares)] -= 1
    while shares.sum() < n_words:
        shares[np.argmin(shares)] += 1

    starts = []
    idx = 0
    for (b_start, b_end), cnt in zip(bursts, shares):
        w_fracs = fracs[idx:idx + cnt]  # boundaries for words in this burst
        # Normalize within-burst: fraction span [w_fracs[0]..w_fracs[-1]]
        lo, hi = w_fracs[0], w_fracs[-1]
        span = (hi - lo) or 1.0
        # Cumulative energy inside this burst (fall back to uniform time).
        bi = np.where((times >= b_start) & (times < b_end))[0]
        if len(bi) >= 2:
            e = env[bi]
            e_cum = np.cumsum(e)
            e_tot = float(e_cum[-1])
        else:
            e_tot = 0.0
        for f in w_fracs:
            rel = (f - lo) / span
            if e_tot > 1e-9:
                k = int(np.searchsorted(e_cum / e_tot, rel, side="left"))
                k = min(max(k, 0), len(bi) - 1)
                t = float(times[bi[k]])
            else:
                t = b_start + rel * (b_end - b_start)
            starts.append(min(t, b_end))
        idx += cnt
    starts[0] = bursts[0][0]

    # Enforce monotonic order with a minimum hold, then build (start, end).
    ends = list(starts[1:]) + [dur]
    for i in range(1, len(starts)):
        if starts[i] < starts[i - 1] + MIN_WORD_GAP:
            starts[i] = starts[i - 1] + MIN_WORD_GAP
            ends[i] = max(ends[i], starts[i])
    ends[-1] = dur
    return list(zip(starts, ends))


def _proportional_timing(words: list, duration: float) -> list:
    """Fallback: no alignment possible; spread words evenly across duration."""
    if duration <= 0:
        duration = len(words) * 0.35
    n = len(words)
    seg = duration / n
    return [(i * seg, (i + 1) * seg) for i in range(n)]


def _part_count(n_words: int) -> int:
    """3 parts normally; 4 for long sentences."""
    return MAX_PARTS if n_words >= LONG_WORD_N else MIN_PARTS


def _split_parts(words: list, n_parts: int) -> list:
    """Split `words` into ~n_parts contiguous parts.

    Boundaries start at even word counts, then nudge to the nearest
    punctuation so parts read like natural spoken phrases.
    """
    n = len(words)
    if n_parts >= n:
        return [[w] for w in words]
    size = math.ceil(n / n_parts)
    bounds = [min(size * p, n) for p in range(1, n_parts)]
    refined = []
    for b in bounds:
        best, best_d = b, abs(b - b)
        for k in range(max(1, b - 2), min(n, b + 3)):
            if words[k - 1][-1] in _PUNCT_PART:
                d = abs(k - b)
                if d < best_d:
                    best_d, best = d, k
        if refined and best <= refined[-1]:
            best = refined[-1] + 1
        if best < n:
            refined.append(best)
    bounds = sorted(refined)
    parts = [words[a:b] for a, b in zip([0] + bounds, bounds + [n])]
    return [p for p in parts if p]


def _layout_words(words: list, font: ImageFont.FreeTypeFont, max_w: int,
                  h: int, w: int) -> list:
    """Lay out a word list as one centered caption block.

    Returns a list of (word, x, y) draw origins in reading order.
    """
    lines = _wrap_words(words, font, max_w)
    line_h = int(font.size * LINE_SPACING)
    block_h = line_h * len(lines)
    top = h - int(h * MARGIN_BOTTOM_FRAC) - block_h
    positions = []
    for ln_idx, ln in enumerate(lines):
        total = font.getlength(" ".join(ln))
        x0 = (w - int(total)) // 2
        cx = x0
        for word in ln:
            positions.append((word, cx, top + ln_idx * line_h))
            cx += font.getlength(word) + font.getlength(" ")
    return positions


def _scale_alpha(overlay: Image.Image, a: float) -> Image.Image:
    """Multiply the overlay's alpha by `a` (0..1) in place and return it."""
    if a >= 1.0:
        return overlay
    a = max(0.0, min(1.0, a))
    alpha = overlay.getchannel("A").point(lambda v: int(v * a))
    overlay.putalpha(alpha)
    return overlay


def add_captions(frames: np.ndarray, text: str, duration: float, accent: str = None,
                 word_times: list = None, frame_rate: float = 24.0) -> np.ndarray:
    """Draw chunked karaoke captions onto each frame in `frames`.

    The sentence is split into 3-4 parts; only the part currently being
    spoken is drawn (karaoke highlight inside it). This is PURELY a visual
    effect -- the audio is never altered by the caption timing.

    frames: uint8 array of shape (T, H, W, 3).
    text:   the line text being spoken (verbatim).
    duration: line duration in seconds.
    accent: hex accent color; None -> DEFAULT_ACCENT.
    word_times: list of (start, end) per word from align_words(); when given,
        the highlighted word is derived from real audio timing (pauses hold).
    frame_rate: playback FPS. Frame i is displayed at t = i/frame_rate, so the
        caption timeline is keyed to wall-clock time (NOT evenly stretched over
        duration -- crossfades drop frames and would desync the highlight).

    Returns a new frame array with captions composited in place of the input.
    """
    text = " ".join((text or "").split())
    if not text:
        return frames

    _, h, w = frames.shape[:3]
    font = _font_for(w)
    words = text.split(" ")
    if word_times is None or len(word_times) != len(words):
        word_times = _proportional_timing(words, duration)

    n_words = len(words)
    max_w = int(w * MAX_WIDTH_FRAC)
    stroke_w = max(1, int(w * STROKE_WIDTH_FRAC))
    shadow_off = max(1, int(h * SHADOW_OFFSET_FRAC))

    # Split into spoken parts + precompute per-part layout and timing window.
    parts = _split_parts(words, _part_count(n_words))
    starts_t = []
    layouts = []
    part_ranges = []  # (first_word_idx, one_past_last_word_idx) in `words`
    gi = 0
    for part in parts:
        s, e = gi, gi + len(part)
        gi = e
        starts_t.append(word_times[s][0])
        layouts.append(_layout_words(part, font, max_w, h, w))
        part_ranges.append((s, e))

    white = (255, 255, 255, 255)
    dim = _hex("#FFFFFF", alpha=110)
    accent_rgba = _hex(accent or DEFAULT_ACCENT)

    n_frames = max(frames.shape[0], 1)
    starts = np.asarray([wt[0] for wt in word_times])
    frame_rate = max(frame_rate, 1e-9)
    out = np.empty_like(frames)

    for i in range(n_frames):
        t_now = i / frame_rate
        # Latest word that has started -> current (holds through pauses).
        current = int(np.searchsorted(starts, t_now, side="right") - 1)
        current = max(0, min(current, n_words - 1))

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)

        if t_now >= starts_t[0]:
            # Active part: the one holding `current`.
            p = 0
            for pi, (ps, pe) in enumerate(part_ranges):
                if ps <= current < pe:
                    p = pi
                    break
            local = current - part_ranges[p][0]
            a = min(1.0, (t_now - starts_t[p]) / PART_FADE_S)
            positions = layouts[p]
            for j, (word, x, y) in enumerate(positions):
                if j < local:
                    fill, stroke = white, (0, 0, 0, 255)
                elif j == local:
                    fill, stroke = accent_rgba, (0, 0, 0, 255)
                else:
                    fill, stroke = dim, (0, 0, 0, 120)
                # Soft drop shadow first (offsets by shadow_off).
                d.text((x + shadow_off, y + shadow_off), word, font=font,
                       fill=(0, 0, 0, 130), stroke_width=stroke_w, stroke_fill=(0, 0, 0, 130))
                d.text((x, y), word, font=font, fill=fill,
                       stroke_width=stroke_w, stroke_fill=stroke)
            if a < 1.0:
                _scale_alpha(overlay, a)

        frame = Image.fromarray(frames[i]).convert("RGBA")
        frame.alpha_composite(overlay)
        out[i] = np.array(frame.convert("RGB"))

    return out
