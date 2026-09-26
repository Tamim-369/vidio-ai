#!/usr/bin/env python3
"""Build a Chatterbox reference from a source recording using the
"prime + filler" strategy:

- The FIRST segment (prime) is a single best continuous clip (~10-12s) that
  leads the reference. Chatterbox conditions prosody/style on the first ~6s
  (LM tokens) and ~10s (diffusion decoder), so this segment is what the voice
  sounds like when talking.
- SEGMENTS AFTER prime only bulk-average into the speaker/timbre embedding, so
  they are additional clean speech runs for identity robustness.

Usage:
    python make_ref.py <voice_id> <source.mp3> [--prime-start 175 --prime-end 185]
        --prime-start/prime-end  if omitted, auto-picks the highest-energy 10s window
        --runs N  number of filler runs after the prime (default 8)
        --out <path> default: candidates/<voice_id>/<voice_id>_ref.wav
        --rms     target loudness (default 0.2)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import soundfile as sf
import librosa


def energy_of(y: np.ndarray, n: int) -> np.ndarray:
    return librosa.feature.rms(y=y, frame_length=n, hop_length=n)[0]


def find_runs(energy: np.ndarray, hop: float, min_dur: float, thr_db: float = 35.0):
    ref_db = float(np.max(energy)) + 1e-9
    above = energy > (ref_db - thr_db)
    runs = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            start, end = i * hop, j * hop
            if (end - start) >= min_dur:
                runs.append((start, end, float(np.mean(energy[i:j]))))
            i = j
        else:
            i += 1
    return runs


def best_prime_window(y: np.ndarray, sr: int, width: float = 10.0) -> tuple:
    win = int(width * sr)
    hop = int(0.25 * sr)
    if len(y) <= win:
        return 0.0, len(y) / sr
    best_r, best_ix = -1.0, 0
    idx = 0
    while idx + win <= len(y):
        r = float(np.sqrt(np.mean(y[idx:idx + win] ** 2)))
        if r > best_r:
            best_r, best_ix = r, idx
        idx += hop
    return best_ix / sr, best_ix / sr + width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice_id", type=str)
    ap.add_argument("source", type=str)
    ap.add_argument("--prime-start", type=float, default=None)
    ap.add_argument("--prime-end", type=float, default=None)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--min-run", type=float, default=2.5)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--rms", type=float, default=0.2)
    ap.add_argument("--start", type=float, default=None, help="ignore source before this (s)")
    ap.add_argument("--end", type=float, default=None, help="ignore source after this (s)")
    args = ap.parse_args()

    src = Path(args.source)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("src/experiments/voices_to_clone/candidates") / args.voice_id / f"{args.voice_id}_ref.wav"

    y, sr = librosa.load(str(src), sr=24000, mono=True)
    dur = len(y) / sr
    if args.start is not None or args.end is not None:
        lo = (args.start or 0.0)
        hi = (args.end or dur)
        lo_i, hi_i = int(lo * sr), max(int(hi * sr), int(lo * sr) + 1)
        y = np.ascontiguousarray(y[lo_i:hi_i])
        dur = len(y) / sr

    # Determine prime clip.
    if args.prime_start is not None and args.prime_end is not None:
        p_start, p_end = args.prime_start, args.prime_end
    else:
        p_start, p_end = best_prime_window(y, sr, width=10.0)
        print(f"Auto-picked highest-energy window: {p_start:.1f}s -> {p_end:.1f}s")

    # Ensure prime within bounds.
    p_end = min(p_end, dur)
    p_start = max(0.0, p_start)
    print(f"Prime clip: {p_start:.1f}s -> {p_end:.1f}s ({p_end - p_start:.1f}s)")

    # Find candidate filler runs (above-threshold speech), avoid overlap with prime and gaps.
    n = int(0.02 * sr)
    energy = energy_of(y, n)
    runs = find_runs(energy, 0.02, min_dur=args.min_run, thr_db=35.0)
    runs.sort(key=lambda r: r[2], reverse=True)

    # Very long runs (continuous speech) mean there are no quiet gaps to segment on;
    # cap them so the window-energy fallback handles picking clean sub-windows.
    MAX_RUN = 8.0
    bounded = [(s, e, m) for s, e, m in runs if (e - s) <= MAX_RUN]

    filler = []
    gap = 0.3
    for s, e, _ in bounded:
        if p_start - gap <= s <= p_end + gap or p_start - gap <= e <= p_end + gap:
            continue  # overlaps the prime
        filler.append((s, e))
        if len(filler) >= args.runs:
            break

    # Continuous speech (no quiet gaps): fall back to top-energy non-overlapping windows.
    if len(filler) < args.runs:
        filler_samples = [(int(s * sr), int(e * sr)) for s, e in filler]
        win = int(4.0 * sr)
        hop = int(0.5 * sr)
        windows = []
        idx = 0
        while idx + win <= len(y):
            r = float(np.sqrt(np.mean(y[idx:idx + win] ** 2)))
            windows.append((r, idx, idx + win))
            idx += hop
        windows.sort(reverse=True)
        for _, ws, we in windows:
            if (ws <= p_start * sr <= we or ws <= p_end * sr <= we
                    or (ws >= p_start * sr and we <= p_end * sr)):
                continue  # overlaps the prime
            if not all(w_end <= ws or w_start >= we for w_start, w_end in filler_samples):
                continue  # overlaps existing filler
            filler_samples.append((ws, we))
            if len(filler_samples) >= args.runs:
                break
        filler = [(s / sr, min(e / sr, dur)) for s, e in filler_samples]
        filler = [(s, e) for s, e in filler if e - s >= args.min_run]

    def norm(seg):
        rms = float(np.sqrt(np.mean(seg ** 2)))
        return seg * (args.rms / (rms + 1e-9)) if rms > 0 else seg

    parts = [norm(y[int(p_start * sr):int(p_end * sr)])]
    labels = [f"prime {p_start:.1f}-{p_end:.1f}s"]
    for s, e in filler:
        parts.append(norm(y[int(s * sr):int(e * sr)]))
        labels.append(f"fill {s:.1f}-{e:.1f}s ({e - s:.1f}s)")

    silence = np.zeros(int(0.15 * sr), dtype=np.float32)
    out = parts[0]
    for p in parts[1:]:
        out = np.concatenate([out, silence, p])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out.astype(np.float32), sr)
    print(f"\nWrote {len(out) / sr:.1f}s reference -> {out_path}")
    for lbl in labels:
        print(f"  {lbl}")
    print(f"Total: {len(parts)} segments ({len(parts) - 1} filler), sample rate {sr}")


if __name__ == "__main__":
    main()