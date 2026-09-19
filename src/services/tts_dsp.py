"""Post-synthesis DSP: pause enforcement, attack dip, finalize, shout handling.

``postprocess_line`` is the single shared seam used by both the live TTS engines
and the harness re-bake path, so pause tweaks reproduce exactly what a fresh
bake would emit.
"""
import os
import re
import shlex

import numpy as np
import soundfile as sf


def _noise_gate(audio: np.ndarray, threshold: float = 0.008) -> np.ndarray:
    """Silence background hiss below threshold WITHOUT the paper-cut clicks.

    Naive ``audio * (envelope > threshold)`` is a BINARY gate: every time the
    envelope crosses threshold on a word onset/offset the multiplier snaps
    0<->1 while the sample is non-zero, producing a tiny "cut a piece of paper"
    click at the START and END of every word. Instead we smooth the gate with
    the SAME 256-sample moving window used to build the envelope, so 0->1 and
    1->0 transitions ramp over ~10 ms instead of snapping. Same window = zero
    phase shift, so word timing is unchanged — only the clicks are removed.
    """
    kernel = np.ones(256) / 256
    envelope = np.convolve(np.abs(audio), kernel, mode='same')
    gate = (envelope > threshold).astype(np.float32)
    gate = np.convolve(gate, kernel, mode='same')  # attack/release smoothing
    return audio * gate


_HOP_S = 0.01                    # RMS envelope hop (s), matches captions timing
_PUNCT_PAUSE = {
    '.': 0.42, '!': 0.42, '?': 0.42,      # sentence end → real breath
    ',': 0.22, ';': 0.25, ':': 0.25,       # reduced from 0.25–0.28 to avoid intruding on speech
    '-': 0.28,
    '\u2013': 0.28, '\u2014': 0.28,
}
_PUNCT_MATCH_W = {                         # max gap-offset (s) for pairing punctuation to a real silence gap
    '.': 0.25, '!': 0.25, '?': 0.25,
    ',': 0.12, ';': 0.12, ':': 0.12,       # commas: tight match only — never force-insert
    '-': 0.12, '\u2013': 0.12, '\u2014': 0.12,
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

    Neural TTS (especially a cloned voice) tends to rush: sentence periods get
    ~0.1 s, commas often nothing. This post-processes so a pause never lands in
    the middle of speech:

      - . ! ?  → at least 0.42 s (a real sentence breath)
      - , ; :  → at least 0.22 s (a beat)
      - dashes → at least 0.28 s

    Word boundaries come from ENERGY ALIGNMENT of this exact audio (the same
    model the captions use), NOT a char-count guess — that guess drifted off by
    up to a second on real Chatterbox output, so punctuation found nothing to
    extend. For each punctuation token:

      1. If a REAL silence the model made overlaps that word boundary, extend
         the gap to the target (never cut into the surrounding speech).
      2. If the model ran the words together (no gap at all), INSERT the pause
         at the aligned boundary — positioned at the quietest frame in a small
         window around the boundary so it lands between words, never mid-word.
         The added pad is pure zeros with 2 ms fades, so no clicks.

    Falls back to the old char-proportional search (lengthen-only, never
    fabricate) when alignment is unavailable or counts don't match the words.
    """
    tokens = (text or "").split()
    n = len(tokens)
    if n < 2 or len(audio) < sr // 2:
        return audio

    dur = len(audio) / sr
    gaps = _detect_silence_gaps(audio, sr)

    # Per-word (start, end) times from the real waveform (same alignment the
    # captions use). Only used to anchor WHERE a boundary is; silences that get
    # lengthened still come from `gaps`.
    times = None
    try:
        from src.services.captions import word_times_from_waveform
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
            have = best_g[1]
            need = target - have
            if need >= 0.04:
                at = best_g[0] + shift  # extend from the gap's start
                out = _insert_silence(out, sr, at, need)
                shift += need
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


def _attack_dip(audio: np.ndarray, sr: int, dip: float = -2.0, window: float = 0.09) -> np.ndarray:
    """Per-word/sentence onset pitch dip — the 'gruff attack' signature.

    Detects rising-energy onsets (word/sentence starts) and briefly drops pitch
    on the attack window via SoX (formant-preserving), crossfading boundaries to
    avoid clicks. dip < 0 makes word starts deeper, gliding back to normal.
    """
    if dip == 0.0:
        return audio
    import librosa
    import subprocess
    import tempfile

    # Onset detection on a 10ms envelope.
    hop = int(sr * 0.01)
    n = int(sr * 0.03)
    env = librosa.feature.rms(y=audio, frame_length=n, hop_length=hop)[0]
    env = env / (env.max() + 1e-9)
    base = float(np.percentile(env, 40))
    thr = base + (1.0 - base) * 0.25

    onsets = []
    above = env > thr
    min_run = int(0.04 / 0.01)  # 40ms sustained energy = a real onset
    min_gap = int(0.15 / 0.01)  # re-trigger only on distinct words/sentences
    prev_end = -10 ** 9
    i = 0
    while i < len(above) - 1:
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if (j - i) >= min_run and i - prev_end >= min_gap:
                onsets.append(i * hop)
                prev_end = j
            i = j
        else:
            i += 1

    if not onsets:
        return audio

    win_len = int(window * sr)
    cents = int(round(dip * 100))
    xf = int(0.008 * sr)  # 8ms crossfade at segment edges to prevent clicks

    # Gather all windows to process, then run sox per window.
    with tempfile.TemporaryDirectory() as td:
        for t0 in onsets:
            s0 = int(t0)
            e0 = min(s0 + win_len, len(audio))
            if e0 - s0 < int(0.03 * sr):
                continue
            seg = audio[s0:e0]
            tmp_in = os.path.join(td, "in.wav")
            tmp_out = os.path.join(td, "out.wav")
            sf.write(tmp_in, seg, sr)
            subprocess.run(
                ["sox", tmp_in, tmp_out, "pitch", str(cents)],
                check=True, capture_output=True,
            )
            seg_p, _ = sf.read(tmp_out, dtype="float32")
            if len(seg_p) != len(seg):
                seg_p = seg_p[:len(seg)]
            # Fade edges (attack inlet / release outlet) so the dip is a glide, not a click.
            f = np.ones(len(seg), dtype=np.float32)
            f[:xf] = np.linspace(0.0, 1.0, xf)
            f[-xf:] = np.linspace(1.0, 0.0, xf)
            seg = seg + (seg_p - seg) * f
            audio[s0:e0] = seg
    return audio


def _onset_boost(audio: np.ndarray, sr: int, peak_gain: float = 2.2, window: float = 0.18) -> np.ndarray:
    """Lift the leading consonant(s) of a line so the first word is clearly audible.

    Chatterbox onsets (especially the first word of a line) come out ~20dB under
    the vowels and attack straight into them, so "Listen" reads as "isten". This
    applies a smooth gain envelope to the first ~180ms of speech: rises 1.0→
    peak_gain over the very first ms (no click — speech is near-silent there),
    holds briefly, then decays back to 1.0. Purely a level lift; timing and
    timbre are untouched.
    """
    if peak_gain <= 1.0 or len(audio) < sr // 4:
        return audio
    n = len(audio)
    hop = int(sr * 0.005)
    frame = audio[:n // hop * hop].reshape(-1, hop)
    rms = np.sqrt(np.mean(frame ** 2, axis=1) + 1e-12)
    # Speech onset = first window clearly above the hiss/hum floor.
    thr = max(rms.max() * 0.05, 0.004)
    idx = int(np.argmax(rms > thr))
    onset = max(idx * hop, 0)
    win = int(window * sr)
    if onset + int(0.004 * sr) >= n:
        return audio
    g = np.ones(n, dtype=np.float32)
    # 4ms attack ramp to peak, then linear decay back to 1.0 across the window.
    attack = max(int(0.004 * sr), 1)
    end = min(onset + win, n)
    seg = np.arange(end - onset, dtype=np.float32)
    decay = 1.0 + (peak_gain - 1.0) * np.clip(1.0 - seg / win, 0.0, 1.0)
    decay[:attack] = np.linspace(1.0, decay[attack], attack)
    g[onset:end] = decay
    return audio * g


def _finalize(combined: np.ndarray, sr: int, pitch_shift: float = 0.0, gain: float = 1.0, eq: list = None, speed: float = 1.0, attack_pitch: float = 0.0, lead_in: float = 0.12) -> np.ndarray:
    """Shared post-processing: EQ + pitch/speed, attack dip, silence pad, noise gate, RMS normalize, clip.

    eq: list of sox filter args (e.g. ["highpass 90", "equalizer 3000 1 2.5"]) applied
    BEFORE normalization so loudness stays constant regardless of the EQ boost.
    speed: playback rate multiplier (<1.0 = slower, >1.0 = faster); pitch preserved.
    attack_pitch: per-onset pitch dip in semitones (negative = gruff word starts).
    lead_in: seconds of clean silence prepended to the clip — the first word of a
    line must never sit at sample 0 (see below); pass a longer value (e.g. 1.0) for
    the very first line of a video so the intro breathes before speech begins.
    """
    if eq or pitch_shift or speed != 1.0:
        # SoX formant-preserving pitch (no phase-vocoder smear like librosa), per-voice
        # timbre EQ and time-stretch. Round-trips through a temp wav since sox CLI works
        # when multiple effects chain.
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_in = os.path.join(td, "in.wav")
            tmp_out = os.path.join(td, "out.wav")
            sf.write(tmp_in, combined, sr)
            effects = []
            if eq:
                for fil in eq:
                    effects.extend(shlex.split(fil))
            if pitch_shift:
                cents = int(round(pitch_shift * 100))
                effects.extend(["pitch", str(cents)])
            if speed != 1.0:
                effects.extend(["tempo", f"{speed:.4f}"])
            subprocess.run(
                ["sox", tmp_in, tmp_out, *effects],
                check=True, capture_output=True,
            )
            combined, _ = sf.read(tmp_out, dtype="float32")

    if attack_pitch:
        combined = _attack_dip(combined, sr, dip=attack_pitch)

    silence = np.zeros(int(0.3 * sr), dtype=combined.dtype)  # 300ms sentence gap
    combined = np.concatenate([combined, silence])
    combined = _noise_gate(combined)

    # Lead-in pad: the first word of a clip must never sit at sample 0. Video
    # encoders/players trim a few tens of ms off the head, which silently eats
    # the (quiet) onset consonant of the very first word ("Listen"→"isten").
    # A short speech-free lead-in protects every line's first phoneme.
    lead_in = np.zeros(int(lead_in * sr), dtype=combined.dtype)
    combined = np.concatenate([lead_in, combined])

    # Onset boost: after the lead-in, the first word's consonant ("Listen"→
    # "isten") is ~20dB under the vowels. Lift the first ~180ms of speech.
    combined = _onset_boost(combined, sr)

    # RMS normalization — consistent loudness across all lines
    TARGET_RMS = 0.15
    rms = np.sqrt(np.mean(combined ** 2))
    if rms > 0:
        combined = combined * (TARGET_RMS / rms)
    if gain != 1.0:
        combined = combined * gain
    combined = np.clip(combined, -0.95, 0.95)
    return combined


# Deterministic emphasis for lines the script marks "loud" (ALL-CAPS / "!!").
# Applied as a simple per-line gain AFTER RMS normalization, so a shout line
# lands consistently louder instead of relying on the TTS model's random
# prosody sampling (which swings the emotion between lines). Kept modest —
# a cloned voice that never screams sounds broken when forced to.
LOUD_GAIN = 1.2


def _de_shout(text: str) -> str:
    """Neutralize ALL-CAPS shouting in the audio text so a cloned voice reads
    naturally (it never yells in its reference audio, so caps make it invent a
    weird delivery). Each all-caps word becomes Title Case, and "!!"/"??"
    collapse to a single mark — the loud flag's volume gain carries the
    emphasis instead. Used for audio synthesis only; captions keep the raw
    shouty line.
    """
    if not text or not any(c.isupper() for c in text):
        return text
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    def _word(m):
        w = m.group(0)
        if w.isupper() and w.isalpha():
            return w.title()
        return w
    return re.sub(r"[A-Za-z']+", _word, text)


def _time_stretch(audio: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Formant-preserving whole-line time-stretch via SoX (factor < 1 = slower,
    factor > 1 = faster).

    Applied across the ENTIRE line uniformly, so it can never create the
    mid-sentence speed step that per-region stretching caused before. The factor
    is CLAMPED to a range that stays artifact-free (so an extremely slow line is
    still picked up as much as is safe rather than skipped entirely).
    """
    if factor <= 0:
        return audio
    factor = min(max(factor, 0.7), 1.5)
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_in = os.path.join(td, "in.wav")
        tmp_out = os.path.join(td, "out.wav")
        sf.write(tmp_in, audio, sr)
        subprocess.run(["sox", tmp_in, tmp_out, "tempo", f"{factor:.4f}"],
                       check=True, capture_output=True)
        out, _ = sf.read(tmp_out, dtype="float32")
    return out


def _strip_lead_buffer(audio: np.ndarray, sr: int, buffer_text: str, full_text: str) -> np.ndarray:
    """Remove a synthesis-only lead word (e.g. 'Okay.') from the front of a clip.

    Chatterbox voices the FIRST phoneme of a fresh synthesis weakly — the very
    cause of "Listen"→"isten". Prepending a throwaway word makes that weak onset
    land on a buffer we then DISCARD: the real first word is synthesized
    mid-stream (after the buffer's pause), where the model voices onsets fully.
    The buffer's trailing pause is detected as the gap after the first speech
    region and cut, so only the real line (plus any pre-roll) remains.
    """
    dur = len(audio) / sr
    if dur < 0.5:
        return audio
    # Threshold tuned for the RAW (pre-finalize, unnormalized) waveform: the
    # buffer's pause can be as short as ~100ms and sits amid model hiss, so the
    # generic env.max()*0.03 gate is too aggressive here and merges the pause
    # into speech. Use the quiet-frame floor instead.
    hop = max(1, int(sr * _HOP_S))
    n = len(audio) // hop
    frame = audio[:n * hop].reshape(n, hop)
    env = np.sqrt(np.mean(frame ** 2, axis=1) + 1e-12)
    quiet = np.percentile(env, 25)
    thr = max(quiet * 1.5, env.max() * 0.02, 0.0025)
    active = env > thr
    gaps = []
    in_gap, start = False, 0
    for i, a in enumerate(active):
        if not a and not in_gap:
            in_gap, start = True, i
        elif a and in_gap:
            in_gap = False
            d = (i - start) * _HOP_S
            if d >= 0.04:
                gaps.append((start * _HOP_S, d))
    if in_gap and (n - start) * _HOP_S >= 0.04:
        gaps.append((start * _HOP_S, (n - start) * _HOP_S))

    # Speech regions are the intervals between gaps.
    regions, cur = [], 0.0
    for gs, gd in gaps:
        if gs - cur >= 0.04:
            regions.append((cur, gs))
        cur = gs + gd
    if dur - cur >= 0.04:
        regions.append((cur, dur))

    # Proportional estimate of where the buffer ends (chars share of duration
    # + half a breath) — used as a sanity bound, never as the exact cut.
    total = max(len(full_text or ""), 1)
    frac = len(buffer_text or "") / total
    est = frac * dur + 0.15

    if len(regions) >= 2:
        # Buffer = first speech region; cut at the start of the second region
        # (end of the buffer's pause). Clamp so we never retain more than
        # ~0.45s of the buffer's trailing breathe even if gap detection is off.
        cut = min(regions[1][0], regions[0][1] + 0.45)
    else:
        cut = est

    print(f"    [tts] Lead-buffer strip: dur={dur:.2f}s regions={len(regions)} est={est:.2f}s cut={cut:.2f}s")
    return audio[int(cut * sr):]


def _normalize_pacing(audio: np.ndarray, sr: int, text: str, params: dict) -> np.ndarray:
    """Equalize speaking rate into the [TTS_MIN_WPS, TTS_MAX_WPS] band.

    The rate is words/second measured over SPEECH-ONLY time — silence gaps are
    subtracted first — so a pause-heavy dramatic line (long breaths between
    staccato words) is not miscounted as slow and its pauses stay intact. Only
    genuine speech tempo is corrected: too-fast lines are slowed to the band
    top, too-slow lines are picked up to the band floor, via a whole-line
    uniform tempo (formant-preserving, no mid-sentence speed step).
    """
    from src.config.settings import TTS_MIN_WPS, TTS_MAX_WPS
    if not TTS_MAX_WPS or TTS_MAX_WPS <= 0:
        return audio
    words = len((text or "").split())
    if words < 2:
        return audio
    dur = len(audio) / sr
    if dur < 0.5:
        return audio
    gaps = _detect_silence_gaps(audio, sr)
    speech = max(dur - sum(d for _, d in gaps), 0.3)
    wps = words / speech
    if wps > TTS_MAX_WPS:
        factor = TTS_MAX_WPS / wps  # slower
    elif wps < TTS_MIN_WPS:
        factor = TTS_MIN_WPS / wps  # faster
    else:
        return audio
    print(f"    [tts] Pacing: {wps:.2f} wps over {speech:.2f}s speech / {dur:.2f}s total -> {factor:.2f}x")
    return _time_stretch(audio, sr, factor)


def postprocess_line(combined: np.ndarray, sr: int, engine: str, text: str, params: dict) -> tuple:
    """Single shared post-synthesis seam (pause enforcement + engine-finalize).

    Used by BOTH the TTS generators and the harness re-bake path so a re-applied
    pause tweak reproduces EXACTLY what a fresh bake would emit — no drift.
    Returns (final_audio, raw_audio) where raw is the pure synthesis
    (pre-pause, pre-finalize) used to rebuild instantly on pause-only changes.
    """
    raw = combined.astype(np.float32)
    final = _enforce_pauses(raw, sr, text)
    if engine == "pocket":
        final = _finalize(final, sr, gain=params.get("gain", 1.0))
    else:  # chatterbox — engine-specific post params
        final = _finalize(
            final, sr,
            pitch_shift=params.get("pitch_shift", 0.0),
            gain=params.get("gain", 1.0),
            eq=params.get("eq"),
            speed=params.get("speed", 1.0),
            attack_pitch=params.get("attack_pitch", 0.0),
        )
    return final, raw
