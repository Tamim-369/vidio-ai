#!/usr/bin/env python3
"""Analyze a source recording and list the longest continuous clean speech runs.

Helps pick a prime ~10-12s segment to lead a Chatterbox reference (the first
~10s is what actually conditions prosody/style) plus filler runs for timbre.

Usage:
    python analyze_source.py <audio.mp3> [--min-dur 2.0] [--top 15]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf
import librosa


def find_runs(energy: np.ndarray, hop: float, min_dur: float, thr_db: float = 35.0):
    """Return (start_s, end_s, mean_rms_db) for continuous above-threshold energy."""
    ref_db = np.max(energy) + 1e-9
    above = energy > (ref_db - thr_db)
    runs = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            start = i * hop
            end = j * hop
            if (end - start) >= min_dur:
                seg = energy[i:j]
                runs.append((start, end, float(np.mean(seg))))
            i = j
        else:
            i += 1
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=str)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--thr-db", type=float, default=35.0)
    args = ap.parse_args()

    y, sr = librosa.load(args.audio, sr=16000, mono=True)
    hop_ms = 20
    hop = hop_ms / 1000.0
    n = int(hop * sr)
    energy = librosa.feature.rms(y=y, frame_length=n, hop_length=n)[0]
    energy_db = 20 * np.log10(energy + 1e-9)

    runs = find_runs(energy_db, hop, args.min_dur, thr_db=args.thr_db)
    if not runs:
        print("No runs found.")
        return

    runs.sort(key=lambda r: r[2], reverse=True)
    print(f"{'rank':>4} {'start_s':>8} {'end_s':>8} {'dur_s':>6} {'mean_db':>8}")
    for i, (s, e, m) in enumerate(runs[: args.top], 1):
        print(f"{i:>4} {s:>8.2f} {e:>8.2f} {e-s:>6.2f} {m:>8.2f}")
    print(f"\nTotal above-threshold speech: {sum(e-s for _, e, _ in runs):.1f}s "
          f"across {len(runs)} runs (>= {args.min_dur}s)")

    longest = max(runs, key=lambda r: r[1] - r[0])
    print(f"Longest continuous run: {longest[0]:.2f}s -> {longest[1]:.2f}s "
          f"({longest[1]-longest[0]:.1f}s)")


if __name__ == "__main__":
    main()