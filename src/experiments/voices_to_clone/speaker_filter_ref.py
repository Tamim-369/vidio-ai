#!/usr/bin/env python3
"""Speaker-cluster a source recording with Chatterbox's own voice encoder,
keep ONLY the target speaker's segments, and rebuild the clone reference.

Why: the identity embedding in Chatterbox's prepare_conditionals pools
skill the ENTIRE reference into one mean vector. If the source contains a
second speaker (interview host, narrator) or heavy noise, their audio is
averaged in too -> the clone sounds like a blend ("robot + <person>") instead
of the person. This tool:

  1. decodes source to mono 24 kHz,
  2. splits into speech runs, chunks them to <= 10 s,
  3. embeds each chunk with Chatterbox's VoiceEncoder (loads ONLY ve.safetensors,
     no 3.5GB TTS model),
  4. clusters embeddings (KMeans, pick k by silhouette over 2..5),
  5. keeps only the TARGET speaker's chunks (majority cluster by default),
  6. picks the best ~10.5s continuous prime window WITHIN that speaker,
  7. appends all remaining chunks of that speaker as identity filler,
  8. writes candidates/<voice_id>/<voice_id>_ref.wav (24 kHz mono).

Usage:
    python src/experiments/voices_to_clone/speaker_filter_ref.py arnold-schwarzenegger \
        src/experiments/voices_to_clone/arnold_voice_to_clone.wav
    python src/experiments/voices_to_clone/speaker_filter_ref.py donald-trump \
        src/experiments/voices_to_clone/trump_voice_to_clone.mp3 --analyze-only
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SR = 24000
VE_SR = 16000               # VoiceEncoder expects 16 kHz; feed it directly to skip resampy
PRIME_W = 10.5
CHUNK = 10.0
MIN_RUN = 1.0


def decode(path: Path, sr: int = SR) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp = tf.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-ac", "1", "-ar", str(sr), tmp], check=True)
        y, got_sr = sf.read(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    if got_sr != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=got_sr, target_sr=sr)
    return y.astype(np.float32)


def load_voice_encoder():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from chatterbox.models.voice_encoder import VoiceEncoder
    ve = VoiceEncoder()
    p = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="ve.safetensors")
    ve.load_state_dict(load_file(p))
    ve.eval()
    return ve


def speech_runs(y: np.ndarray):
    runs = librosa.effects.split(y, top_db=26, frame_length=1024, hop_length=256)
    return [(int(a), int(b)) for a, b in runs if (b - a) / SR >= MIN_RUN]


def chunk_runs(runs, max_s=CHUNK):
    """(a, b) sample-index windows <= max_s seconds covering each run."""
    out = []
    max_n = int(max_s * SR)
    for a, b in runs:
        L = b - a
        if L <= max_n:
            out.append((a, b))
        else:
            # non-overlapping 10s chunks (slight overlap not needed for clustering)
            i = a
            while i < b:
                j = min(i + max_n, b)
                if (j - i) / SR >= MIN_RUN:
                    out.append((i, j))
                i = j
    return out


def prime_window_score(y: np.ndarray, a: int, b: int, env_db: np.ndarray,
                       active_db: float, frame_t: np.ndarray, flat: np.ndarray):
    """Score a candidate window: loud, stable, harmonic, continuous speech."""
    hop = 256
    i0, i1 = int(a / hop), int(b / hop)
    if i1 - i0 < 10:
        return float("-inf")
    seg_db = env_db[i0:i1]
    seg_flat = np.interp(frame_t[i0:i1], np.arange(len(flat)) * (512 / SR), flat)
    sp = seg_db > active_db
    if sp.sum() < len(seg_db) * 0.4:
        return float("-inf")
    return (float(np.mean(seg_db[sp]))
            - 0.45 * float(np.std(seg_db[sp]))
            - float(np.median(seg_flat[sp])) * 10.0
            - 45.0 * (1.0 - sp.sum() / len(seg_db)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice_id")
    ap.add_argument("source")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", default="majority",
                    choices=["majority", "centroid-close"],
                    help="which cluster = target speaker")
    ap.add_argument("--min-keep-dur", type=float, default=10.0,
                    help="skip clusters with < this many seconds (probably noise/host)")
    ap.add_argument("--analyze-only", action="store_true",
                    help="just report clusters, do not rewrite the ref")
    ap.add_argument("--rms", type=float, default=0.2)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        Path("src/experiments/voices_to_clone/candidates") / args.voice_id / f"{args.voice_id}_ref.wav")

    y = decode(Path(args.source))
    print(f"Source: {len(y)/SR:.1f}s @ {SR} Hz mono")
    runs = speech_runs(y)
    chunks = chunk_runs(runs)
    print(f"Runs: {len(runs)}, chunks: {len(chunks)} (<= {CHUNK}s each)")

    print("Loading voice encoder (small, no T3/S3Gen)...", flush=True)
    ve = load_voice_encoder()

    # VE needs 16 kHz; resample a copy (default librosa resample - no resampy needed)
    y16 = librosa.resample(y, orig_sr=SR, target_sr=VE_SR).astype(np.float32)
    segs16 = [y16[int(a / SR * VE_SR):int(b / SR * VE_SR)] for a, b in chunks]
    embs = ve.embeds_from_wavs(segs16, sample_rate=VE_SR)  # (N, E)
    embs = np.asarray(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs_n = embs / norms

    # --- choose k by silhouette over 2..5 ---
    best_k, best_sil = 2, -1.0
    for k in range(2, 6):
        if len(embs_n) < k + 1:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(embs_n)
        sil = silhouette_score(embs_n, labels)
        if sil > best_sil:
            best_k, best_sil = k, sil
        print(f"  k={k}: silhouette={sil:.3f}")
    km = KMeans(n_clusters=best_k, n_init=10, random_state=0)
    labels = km.fit_predict(embs_n)
    print(f"Best k={best_k} (silhouette {best_sil:.3f})")

    # --- per-cluster duration ---
    durs = np.array([(b - a) / SR for a, b in chunks])
    total = durs.sum()
    print("\nCluster breakdown (target speaker stays):")
    cluster_ids = []
    for c in range(best_k):
        m = labels == c
        c_dur = durs[m].sum()
        cluster_ids.append((c, c_dur, m.sum()))
        print(f"  cluster {c}: {c_dur:6.1f}s ({c_dur/total*100:4.1f}%), {int(m.sum())} chunks, "
              f"emb_norm={float(np.linalg.norm(embs[m].mean(0))):.3f}")
    cluster_ids.sort(key=lambda x: x[1], reverse=True)
    target_c = cluster_ids[0][0]  # majority

    # pick prime window (best continuous speech) -- informational
    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    env_db = 20 * np.log10(rms + 1e-12)
    frame_t = np.arange(len(env_db)) * hop / SR
    floor_db = float(np.percentile(env_db, 15))
    active_db = floor_db + 12.0
    flat, _ = librosa.effects.trim(y, top_db=20) if False else (None, None)
    st = librosa.stft(y, n_fft=2048, hop_length=512, window="hann")
    p = np.abs(st) ** 2
    flat = np.exp(np.mean(np.log(p + 1e-12), axis=0)) / (np.mean(p, axis=0) + 1e-12)
    flat_t = np.arange(len(flat)) * (512 / SR)

    # best window restricted to target-cluster chunks
    target_chunks = [chunks[i] for i in np.where(labels == target_c)[0]]
    best = None
    w = int(PRIME_W * SR)
    step = int(0.25 * SR)
    for a, b in target_chunks:
        s = a
        while s + w <= b:
            sc = prime_window_score(y, s, s + w, env_db, active_db, frame_t, flat)
            if best is None or sc > best[0]:
                best = (sc, s, s + w)
            s += step
    print(f"\nPrime (target cluster): {best[1]/SR:.1f}s -> {best[2]/SR:.1f}s score={best[0]:.1f}")

    if args.analyze_only:
        print("\n--analyze-only: not writing ref.")
        return

    # --- build ref from target-cluster chunks only ---
    keep_idx = [i for i in np.where(labels == target_c)[0]]
    # merge adjacent chunks back into (a,b) intervals
    keep_intervals = sorted(chunks[i] for i in keep_idx)
    merged = []
    for a, b in keep_intervals:
        if merged and a - merged[-1][1] <= int(0.35 * SR):
            merged[-1][1] = b
        else:
            merged.append([a, b])

    prime_a, prime_b = best[1], best[2]
    parts = []
    # prime
    prime = y[prime_a:prime_b]
    prime, _ = librosa.effects.trim(prime, top_db=22)
    parts.append(prime.astype(np.float32))

    # filler: everything in merged intervals outside the prime
    for a, b in merged:
        # clip out prime overlap
        segs_out = []
        if prime_a < b and prime_b > a:  # overlaps
            if a < prime_a:
                segs_out.append((a, prime_a))
            if b > prime_b:
                segs_out.append((prime_b, b))
        else:
            segs_out.append((a, b))
        for sa, sb in segs_out:
            if (sb - sa) / SR < 0.7:
                continue
            seg = y[sa:sb]
            seg, _ = librosa.effects.trim(seg, top_db=20)
            if len(seg) < int(0.7 * SR):
                continue
            r = float(np.sqrt(np.mean(seg ** 2)))
            if r > 0:
                parts.append((seg * (args.rms / r)).astype(np.float32))

    silence = np.zeros(int(0.15 * SR), dtype=np.float32)
    out = parts[0]
    for seg in parts[1:]:
        out = np.concatenate([out, silence, seg])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), (out * (args.rms / (np.sqrt(np.mean(out ** 2)) + 1e-12))).astype(np.float32), SR)
    print(f"\nWrote ref: {len(out)/SR:.1f}s  ({len(parts)} parts) -> {out_path}")


if __name__ == "__main__":
    main()
