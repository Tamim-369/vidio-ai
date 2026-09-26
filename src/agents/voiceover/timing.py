"""Word-level timing from a narration waveform.

The card shows a whole line at once, so nothing is burned in as subtitles. Word
timings still matter: dsp._enforce_pauses() uses them to place a pause at a real
sentence boundary instead of an arbitrary offset.

Audio-forced rather than proportional: the waveform is split into bursts at real
silences and each word starts by cumulative speech energy inside its burst, so
long or loud words get more time and a word holds through a pause."""

import numpy as np

# Word-timing (audio-forced alignment) constants.
HOP_S = 0.01                   # RMS envelope hop in seconds
SILENCE_RATIO = 0.025          # threshold = peak * this (activity gate)
MIN_WORD_GAP = 0.045           # force a word to hold at least this long (s)


def _proportional_timing(words: list, duration: float) -> list:
    """Fallback: no alignment possible; spread words evenly across duration.

    n = len(words) with no guard: an empty word list divides by zero. That is a
    known, long-standing behaviour pinned by test_captions; the callers all
    return early on empty text, so it is not reachable from the render path.
    """
    if duration <= 0:
        duration = len(words) * 0.35
    n = len(words)
    seg = duration / n
    return [(i * seg, (i + 1) * seg) for i in range(n)]


def word_times_from_waveform(y: np.ndarray, sr: int, text: str) -> list:
    """Core word-timing model, run on in-memory audio.

    Splits the waveform into speech bursts at real silences, then distributes
    each word's start by cumulative speech energy weighted by word size.
    Returns a list of (start_s, end_s) per word.
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
