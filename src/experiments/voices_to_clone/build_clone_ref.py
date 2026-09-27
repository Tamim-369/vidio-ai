#!/usr/bin/env python3
"""Build a Chatterbox clone reference from a raw recording, from scratch.

How Chatterbox consumes the reference (verified in its source):
  - first ~6s  (ENC_COND_LEN) -> LM conditioning tokens  (style / prosody / delivery)
  - first ~10s (DEC_COND_LEN) -> S3Gen decoder conditioning (timbre conditioning on the mel)
  - EVERYTHING ELSE           -> voice-encoder speaker embedding  (identity, chunked + mean-pooled)

So we:
  1. pick the single best ~10-11s continuous window (the PRIME) -> leads the file,
     drives prosody and decoder conditioning;
  2. append AS MUCH of the remaining clean speech as possible -> feeds the identity
     embedding. Obvious non-speech (pure crowd/applause/noise) is skipped unless
     --keep-all is passed.

Usage:
    python src/experiments/voices_to_clone/build_clone_ref.py donald-trump src/experiments/voices_to_clone/trump_voice_to_clone.mp3
    python src/experiments/voices_to_clone/build_clone_ref.py arnold-schwarzenegger src/experiments/voices_to_clone/arnold_voice_to_clone.wav --prime-start 0 --prime-end 0
    python src/experiments/voices_to_clone/build_clone_ref.py <voice_id> <source> --out custom.wav --keep-all
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

SR = 24000          # reference sample rate (matches chatterbox S3GEN_SR / old refs)
PRIME_W = 10.5      # seconds of leading prime window
FRAME_N = 1024
FRAME_H = 256


def decode(path: Path, sr: int = SR):
    """Decode any file to mono float32 at `sr` via ffmpeg (handles mp3/wav/video)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp = tf.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1",
             "-ar", str(sr), tmp], check=True)
        y, got_sr = sf.read(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    if got_sr != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=got_sr, target_sr=sr)
    return y.astype(np.float32)


def _stft_features(y: np.ndarray, n=2048, hop=512):
    st = librosa.stft(y, n_fft=n, hop_length=hop, window="hann")
    p = np.abs(st) ** 2
    log_p = np.log(p + 1e-12)
    geo = np.exp(np.mean(log_p, axis=0))
    arith = np.mean(p, axis=0)
    flat = np.clip(geo / (arith + 1e-12), 0.0, 1.0)
    t = np.arange(len(flat)) * hop / SR
    return flat, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice_id", type=str)
    ap.add_argument("source", type=str)
    ap.add_argument("--out", type=str, default=None,
                    help="output path (default candidates/<voice_id>/<voice_id>_ref.wav)")
    ap.add_argument("--prime-start", type=float, default=None,
                    help="force prime window start (s); overrides auto-pick")
    ap.add_argument("--prime-end", type=float, default=None,
                    help="force prime window end (s); must be given with --prime-start")
    ap.add_argument("--min-part", type=float, default=1.2,
                    help="minimum filler segment length in seconds (quiet slices are noise boosters)")
    ap.add_argument("--min-rms", type=float, default=0.02,
                    help="skip filler whose post-trim RMS is below this (would need >20dB boost -> shrill)")
    ap.add_argument("--max-parts", type=int, default=40,
                    help="cap on number of filler segments kept (quality over quantity)")
    ap.add_argument("--rms", type=float, default=0.2, help="target RMS of each part")
    ap.add_argument("--keep-all", action="store_true",
                    help="never drop noise-like segments (crowd/applause) from filler")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        Path("src/experiments/voices_to_clone/candidates") / args.voice_id / f"{args.voice_id}_ref.wav")

    print(f"Decoding {args.source} ...", flush=True)
    y = decode(Path(args.source))
    dur = len(y) / SR
    print(f"  {dur:.1f}s, {SR} Hz mono")

    # ---- speech activity + spectral flatness envelopes ----
    hop = FRAME_H
    rms = librosa.feature.rms(y=y, frame_length=FRAME_N, hop_length=hop)[0]
    env_db = 20 * np.log10(rms + 1e-12)
    frame_t = np.arange(len(env_db)) * hop / SR

    speech_db = env_db[env_db > np.percentile(env_db, 15)]
    floor_db = float(np.percentile(env_db, 15))
    active_db = floor_db + 12.0
    print(f"  noise floor ~{floor_db:.1f} dBFS, speech gate ~{active_db:.1f} dBFS")

    flat, flat_t = _stft_features(y)
    flat_at = np.interp(frame_t, flat_t, flat)

    # ---- prime window: best stable, loud, harmonic ~10.5s of speech ----
    if args.prime_start is not None:
        assert args.prime_end is not None, "--prime-end required with --prime-start"
        p_s, p_e = float(args.prime_start), float(args.prime_end)
        p_i0, p_i1 = int(p_s * SR), min(int(p_e * SR), len(y))
        print(f"  forced prime: {p_s:.1f}s -> {p_e:.1f}s")
    else:
        width = PRIME_W
        win = int(width / (hop / SR))
        scores = []
        step = max(1, int(0.25 / (hop / SR)))
        for i in range(0, len(env_db) - win, step):
            seg_db = env_db[i:i + win]
            seg_flat = flat_at[i:i + win]
            sp = seg_db > active_db
            if sp.sum() < win * 0.4:
                scores.append((float("-inf"), i, seg_db, sp))
                continue
            mean_db = float(np.mean(seg_db[sp]))
            std_db = float(np.std(seg_db[sp]))
            sil_frac = 1.0 - sp.sum() / win
            flat_pen = float(np.median(seg_flat[sp])) * 10.0
            score = mean_db - 0.45 * std_db - flat_pen - 45.0 * sil_frac
            scores.append((score, i, seg_db, sp))
        scores.sort(reverse=True, key=lambda s: s[0])
        score, i0, _, _ = scores[0]
        p_i0, p_i1 = i0 * hop, (i0 + win) * hop
        p_s, p_e = p_i0 / SR, p_i1 / SR
        print(f"  auto-picked prime: {p_s:.1f}s -> {p_e:.1f}s (score {score:.1f})")
        for nxt, iN, _, _ in scores[1:4]:
            if nxt > float("-inf"):
                print(f"    runner-up: {iN*hop/SR:.1f}s -> {(iN+win)*hop/SR:.1f}s ({nxt:.1f})")

    # trim leading/trailing silence from the prime window itself
    prime = y[p_i0:p_i1]
    prime, _pi = librosa.effects.trim(prime, top_db=22)
    prime = prime.astype(np.float32)
    if len(prime) < int(6 * SR):
        print("  ⚠️  prime shorter than 6s after silence-trim")
    p_start = p_i0 / SR
    print(f"  prime after trim: {p_start:.1f}s -> {p_start + len(prime)/SR:.1f}s ({len(prime)/SR:.1f}s)")

    # ---- filler: all remaining speech runs, as much as possible ----
    keep_mask = np.ones(len(y), dtype=bool)
    keep_mask[p_i0:p_i0 + len(prime)] = False

    print("\nSegmenting remaining speech ...")
    runs = librosa.effects.split(y, top_db=26, frame_length=FRAME_N, hop_length=hop)
    parts = [("prime", prime)]
    skipped, kept = [], []

    def trim_and_norm(seg):
        seg, _idx = librosa.effects.trim(seg, top_db=20)
        if len(seg) < int(args.min_part * SR):
            return None
        r = float(np.sqrt(np.mean(seg ** 2)))
        if r <= 0 or r < args.min_rms:
            return None
        return seg * (args.rms / r)

    for (a, b) in runs:
        if b - a < int(args.min_part * SR):
            continue
        # clip this run to the region outside the prime
        seg = y[a:b]
        if not keep_mask[a:b].all():
            # interior overlap with prime -> take only the outside slices
            ov = ~keep_mask[a:b]
            idx = np.where(ov)[0]
            # handle simple cases: prime at left edge, right edge, or covering whole run
            first, last = idx[0], idx[-1]
            kept1 = seg[:first] if first > int(args.min_part * SR) else None
            kept2 = seg[last + 1:] if len(seg) - last - 1 > int(args.min_part * SR) else None
            for kp in (kept1, kept2):
                if kp is not None:
                    s = trim_and_norm(kp)
                    if s is not None:
                        parts.append(("fill", s))
            continue
        s = trim_and_norm(seg)
        if s is None:
            continue
        parts.append(("fill", s))

    # merge contiguous parts? no - keep parts, join with 150ms silence
    if len(parts) > 1:
        labelled = parts[1:]
        flat_med = np.median(flat_at)
        print(f"  {len(labelled)} filler segments found")
        if not args.keep_all:
            # drop likely crowd/applause/music-only segments (high spectral flatness)
            clean, noisy = [], []
            for _, seg in labelled:
                f, _ = _stft_features(seg)
                fm = float(np.median(f))
                (noisy if fm > 0.65 else clean).append((_, seg, fm))
            print(f"  kept {len(clean)} clean segment(s), skipped {len(noisy)} noise-like; "
                  f"pass --keep-all to include them")
            for _, seg, fm in noisy:
                print(f"    skipped {len(seg)/SR:5.1f}s  flatness={fm:.2f}")
            labelled = clean
        else:
            labelled = [(kind, seg, None) for kind, seg in labelled]
        parts = [parts[0]] + [(k, s) for k, s, _ in labelled]

    # ---- assemble ----
    silence = np.zeros(int(0.15 * SR), dtype=np.float32)
    out = parts[0][1]
    filler_total = 0.0
    for kind, seg in parts[1:]:
        out = np.concatenate([out, silence, seg])
        filler_total += len(seg) / SR

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), (out * (args.rms / (np.sqrt(np.mean(out ** 2)) + 1e-12))).astype(np.float32), SR)

    print(f"\nRef total: {len(out)/SR:.1f}s  (prime {len(parts[0][1])/SR:.1f}s + {filler_total:.1f}s filler, "
          f"{len(parts)-1} segments)")
    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()