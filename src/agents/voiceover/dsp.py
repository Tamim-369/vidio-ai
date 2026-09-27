"""Post-synthesis DSP: pause enforcement, attack dip, finalize, shout handling.

postprocess_line() is the shared seam used by both the live TTS engines and the
harness re-bake path, so a pause tweak reproduces what a fresh bake emits.
"""
import os
import re
import shlex

import numpy as np
import soundfile as sf


def _noise_gate(audio: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Silence background hiss below threshold. A window near the threshold is
    ramped rather than zeroed, so the cut is a fade instead of a click."""
    kernel = np.ones(256) / 256
    envelope = np.convolve(np.abs(audio), kernel, mode='same')
    if threshold is None:
        peak = float(envelope.max())
        # Ignore true silence (model pauses, our zero pads): otherwise the
        # percentile lands at 0 and the threshold collapses to ~peak*0.008.
        non_silent = envelope[envelope > peak * 0.002]
        if len(non_silent):
            floor = float(np.percentile(non_silent, 20))  # Lower percentile = more conservative
        else:
            floor = peak * 0.01
        threshold = max(floor * 1.8, peak * 0.006)  # Higher multiplier = stricter gate
        threshold = min(threshold, peak * 0.06)  # Lower ceiling = never catch speech
    gate = (envelope > threshold).astype(np.float32)
    gate = np.convolve(gate, kernel, mode='same')  # attack/release smoothing
    return audio * gate


from src.agents.voiceover.pauses import _detect_silence_gaps, _enforce_pauses


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


def _finalize(combined: np.ndarray, sr: int, pitch_shift: float = 0.0, gain: float = 1.0, eq: list = None, speed: float = 1.0, attack_pitch: float = 0.0, lead_in: float = 0.06) -> np.ndarray:
    """Shared post-processing: EQ + pitch/speed, attack dip, silence pad, noise gate, RMS normalize, clip.

    eq: list of sox filter args (e.g. ["highpass 90", "equalizer 3000 1 2.5"]) applied
    BEFORE normalization so loudness stays constant regardless of the EQ boost.
    speed: playback rate multiplier (<1.0 = slower, >1.0 = faster); pitch preserved.
    attack_pitch: per-onset pitch dip in semitones (negative = gruff word starts).
    lead_in: seconds of clean silence prepended to the clip — the first word of a
    line must never sit at sample 0 (see below). Keep SHORT: a large pad reads as
    dead air before every line and makes the video feel laggy.
    """
    if eq or pitch_shift or speed != 1.0:
        # SoX formant-preserving pitch (no phase-vocoder smear like librosa),
        # per-voice timbre EQ and time-stretch. Round-trips via a temp wav since
        # the sox CLI needs one when effects chain.
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

    silence = np.zeros(int(0.06 * sr), dtype=combined.dtype)  # 60ms line tail — no dead air
    combined = np.concatenate([combined, silence])
    combined = _noise_gate(combined)

    # Lead-in pad: the first word must never sit at sample 0. Encoders trim a
    # few tens of ms off the head, which eats the quiet onset consonant
    # ("Listen"->"isten"). Keep it short or it becomes dead air on every line.
    lead_in = np.zeros(int(lead_in * sr), dtype=combined.dtype)
    combined = np.concatenate([lead_in, combined])

    # Onset boost: the first consonant sits ~20dB under the vowels.
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
# A per-line gain AFTER RMS normalization, so a shout lands consistently louder
# instead of relying on the TTS model's random prosody sampling.
LOUD_GAIN = 1.2

# Speaking-rate band in words per second, measured over speech-only time (silence
# subtracted) so deliberate pauses are not punished. Only lines whose speech
# bursts fall outside the band get a whole-line uniform tempo correction. Tuned
# to natural energised narration: a 2.6 floor left lines audibly draggy and a
# 1.05x stretch felt robotic, so the floor sits at normal narration (+1) and the
# ceiling reins in the rare rush. TTS_MAX_WPS=0 disables.
TTS_MIN_WPS = float(os.getenv("TTS_MIN_WPS", "3.0"))
TTS_MAX_WPS = float(os.getenv("TTS_MAX_WPS", "3.4"))


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
    still picked up as much as is safe rather than skipped entirely). Never
    slows more than 3%: anything below 0.97 reads as a sudden speed drop.
    """
    if factor <= 0:
        return audio
    factor = min(max(factor, 0.97), 1.5)
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


def _normalize_pacing(audio: np.ndarray, sr: int, text: str, params: dict, is_first: bool = False) -> np.ndarray:
    """Equalize speaking rate into the [TTS_MIN_WPS, TTS_MAX_WPS] band.

Words/second over SPEECH-ONLY time (silence subtracted), so a pause-heavy line
is not miscounted as slow and its pauses stay intact. Correction is a
whole-line uniform tempo (formant-preserving, no mid-sentence step).

is_first=True enforces exact mid-band pace so the opening sounds normal."""
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
    
    # Mid-band target for "normal" pace
    mid_wps = (TTS_MIN_WPS + TTS_MAX_WPS) / 2.0  # 3.2
    
    if is_first:
        # First line: enforce EXACT mid-band pace
        if abs(wps - mid_wps) > 0.05:  # only adjust if meaningfully off
            factor = mid_wps / wps
            # Cap at reasonable bounds to avoid artifacts
            factor = min(max(factor, 0.85), 1.15)
            print(f"    [tts] First-line pacing: {wps:.2f} -> {mid_wps:.2f} wps ({factor:.2f}x)")
            return _time_stretch(audio, sr, factor)
        return audio
    
    if wps > TTS_MAX_WPS:
        # Cap the slow-down at 3%: a line that ran hot must not suddenly gum up
        # against the band ceiling (perceptible as a mid-video speed drop).
        factor = max(TTS_MAX_WPS / wps, 0.97)
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
