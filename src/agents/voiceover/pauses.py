"""Pause enforcement: turn sentence punctuation into real silence.

Kept apart from the tone/finalize chain in dsp.py because this half is about
*where* silence goes (text -> gap placement) rather than how a clip sounds.
"""

import numpy as np

_HOP_S = 0.01                    # RMS envelope hop (s), matches captions timing
_PUNCT_PAUSE = {
    '.': 0.18, '!': 0.18, '?': 0.18,       # sentence end → short beat
    ',': 0.10, ';': 0.12, ':': 0.12,
    '-': 0.12,
    '\u2013': 0.12, '\u2014': 0.12,
}
_PUNCT_MATCH_W = {                         # max gap-offset (s) for pairing punctuation to a real silence gap
    '.': 0.20, '!': 0.20, '?': 0.20,
    ',': 0.10, ';': 0.10, ':': 0.10,       # commas: tight match only — never force-insert
    '-': 0.10, '\u2013': 0.10, '\u2014': 0.10,
}
_TRAILING_QUOTES = "'\"\u2019\u201d\u201c"


def _trailing_punct(token: str):
    """The sentence/clause punctuation a token ends with (None if none).

    Strips quotes/brackets first: ['"Total."', '"war,"'] → '.', ','.
    Decimal/range numbers are safe — their trailing char is a digit.
    """
    t = token
    while t and t[-1] in _TRAILING_QUOTES:
        t = t[:-1]
    for ch in ('.', '!', '?', ',', ';', ':', '-', '\u2013', '\u2014'):
        if t.endswith(ch):
            return ch
    return None


def _detect_silence_gaps(audio: np.ndarray, sr: int) -> list:
    """(start_s, dur_s) of continuous quiet regions >= 60 ms, from RMS energy."""
    hop = max(1, int(sr * _HOP_S))
    n = len(audio) // hop
    if n < 2:
        return []
    frame = audio[:n * hop].reshape(n, hop)
    env = np.sqrt(np.mean(frame ** 2, axis=1) + 1e-12)
    thr = env.max() * 0.03
    active = env > thr
    gaps = []
    in_gap, start = False, 0
    for i, a in enumerate(active):
        if not a and not in_gap:
            in_gap, start = True, i
        elif a and in_gap:
            in_gap = False
            d = (i - start) * _HOP_S
            if d >= 0.06:
                gaps.append((start * _HOP_S, d))
    if in_gap and (n - start) * _HOP_S >= 0.06:
        gaps.append((start * _HOP_S, (n - start) * _HOP_S))
    return gaps


def _insert_silence(audio: np.ndarray, sr: int, at_s: float, dur_s: float) -> np.ndarray:
    """Insert `dur_s` seconds of pure silence just before timestamp `at_s`.

    The added pad is ONLY zeros — the splice edges of the surrounding audio get
    a short (2 ms) fade so the cut never clicks. (A previous version faded the
    pad itself up/down, which created an audible "paper-cut" click inside every
    pause.)
    """
    idx = min(max(int(at_s * sr), 0), len(audio))
    left = audio[:idx].copy()
    right = audio[idx:].copy()
    fade = min(int(0.002 * sr), len(left), len(right))
    if fade:
        left[-fade:] *= np.linspace(1.0, 0.0, fade)
        right[:fade] *= np.linspace(0.0, 1.0, fade)
    pad = np.zeros(int(dur_s * sr), dtype=audio.dtype)
    return np.concatenate([left, pad, right])


def _enforce_pauses(audio: np.ndarray, sr: int, text: str) -> np.ndarray:
    """Give the audio the pauses the punctuation implies.

Neural TTS tends to rush: periods get ~0.1s, commas often nothing. Targets are
0.20s for . ! ?, 0.12-0.14s for , ; :, 0.15s for dashes.

Boundaries come from energy alignment of this exact audio, not a char-count
guess (that drifted up to a second on real Chatterbox output). Per punctuation
token, either bring a real overlapping silence to the target - lengthening, or
TRIMMING an over-long breath - or, if the words ran together with no gap,
INSERT the pause at the quietest frame near the boundary so it never lands
mid-word. Added pad is pure zeros with 2ms fades, so no clicks.

Falls back to a char-proportional search (lengthen-only, never fabricate) when
alignment is unavailable or the counts do not match the words."""
    tokens = (text or "").split()
    n = len(tokens)
    if n < 2 or len(audio) < sr // 2:
        return audio

    dur = len(audio) / sr
    gaps = _detect_silence_gaps(audio, sr)

    # Per-word (start, end) times from the real waveform. Only used to anchor
    # WHERE a boundary is; the silence length itself still comes from `gaps`.
    times = None
    try:
        from src.agents.voiceover.timing import word_times_from_waveform
        times = word_times_from_waveform(audio, sr, text)
        if len(times) != n:
            times = None
    except Exception:
        times = None

    # Char-proportional fallback (lengthen-only, conservative).
    charlen = np.cumsum([len(t) + 1 for t in tokens])
    total = float(charlen[-1])

    # Envelope used to pick the quietest frame near a boundary (for insertion).
    hop = max(1, int(sr * _HOP_S))
    nf = len(audio) // hop
    if nf > 0:
        frame = audio[:nf * hop].reshape(nf, hop)
        env = np.sqrt(np.mean(frame ** 2, axis=1) + 1e-12)
    else:
        env = np.zeros(0)

    def _quietest_time(center: float) -> float:
        """Time of the lowest-energy frame within +-0.035s of `center`."""
        i0 = max(int((center - 0.035) / _HOP_S), 0)
        i1 = min(int((center + 0.035) / _HOP_S) + 1, len(env))
        if i1 <= i0 or i0 >= len(env):
            return max(min(center, dur - 0.01), 0.0)
        k = i0 + int(np.argmin(env[i0:i1]))
        return min(k * _HOP_S, dur - 0.01)

    out = audio
    shift = 0.0  # cumulative inserted time this line
    for i in range(n - 1):
        token = tokens[i]
        punct = _trailing_punct(token)
        if not punct:
            continue
        target = _PUNCT_PAUSE[punct]

        if times is not None:
            # Aligned boundary: the end of this word / start of the next word.
            w_end = times[i][1]
            n_start = times[i + 1][0]
            lo = max(w_end - 0.05, 0.0)
            hi = min(n_start + 0.02, dur)
        else:
            mat_ch = _PUNCT_MATCH_W[punct]
            est = charlen[i] / total * dur
            lo, hi = est - mat_ch, est + mat_ch

        # 1) Prefer lengthening a REAL silence the model made near the boundary.
        best_g, best_d = None, 1e9
        for (gs, gd) in gaps:
            if gs >= hi or gs + gd <= lo:
                continue
            center = gs + gd / 2
            if lo <= center <= hi:
                d = 0.0
            else:
                d = min(abs(center - lo), abs(center - hi))
            if d < best_d:
                best_d, best_g = d, (gs, gd)
        if best_g is not None:
            gs, have = best_g
            if have > target + 0.05:  # More tolerant of natural pauses
                # Model left an over-long breath: dead audio + hiss fog between
                # sentences. Replace the WHOLE gap with `target` clean zeros —
                # trims the dead air AND zeros out the foggy noise at once.
                i0 = int((gs + shift) * sr)
                i1 = int(((gs + have) + shift) * sr)
                i1 = min(i1, len(out))
                left = out[:i0].copy()
                right = out[i1:].copy()
                fade = min(int(0.002 * sr), len(left), len(right))
                if fade:
                    left[-fade:] *= np.linspace(1.0, 0.0, fade)
                    right[:fade] *= np.linspace(0.0, 1.0, fade)
                pad = np.zeros(int(target * sr), dtype=out.dtype)
                out = np.concatenate([left, pad, right])
                shift += target - have
            elif target - have >= 0.05:  # Only extend if meaningfully short
                at = gs + shift  # extend from the gap's start
                out = _insert_silence(out, sr, at, target - have)
                shift += target - have
            continue

        # 2) Model ran words together → INSERT the pause at the aligned
        #    boundary, at its quietest frame. Only when we have real alignment
        #    (a char-count guess is exactly what caused the mid-word bug).
        if times is None:
            continue
        at = _quietest_time(w_end) + shift
        pad = target
        if pad < 0.04 or at + pad * sr > len(out):
            continue
        out = _insert_silence(out, sr, at, pad)
        shift += pad
    return out
