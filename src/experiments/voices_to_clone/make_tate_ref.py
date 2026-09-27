"""Tate-specific ref builder. Deliberately separate from the Arnold/Trump tooling.

Builds a chatterbox ref from a calm-register selection of the Tate source.
Usage:
    make_tate_ref.py --prime 298.6 --out candidates/andrew-tate/andrew-tate_ref.wav
    make_tate_ref.py --prime 103.5 --out candidates/andrew-tate/andrew-tate_ref_b.wav
"""

import argparse
import numpy as np
import librosa
import soundfile as sf

SR = 24000
HOP = 256
W = 10.5


def analyze(y, a, b):
    seg = y[int(a * SR):int(b * SR)]
    rms = np.sqrt(np.mean(seg**2))
    f0, v, _ = librosa.pyin(seg, fmin=55, fmax=400, sr=SR, frame_length=2048, hop_length=512)
    idx = np.where(v)[0]
    f0v = f0[idx]
    med = float(np.median(f0v)) if len(idx) > 5 else 0.0
    sd = float(np.std(f0v)) if len(idx) > 5 else 999.0
    cent = librosa.feature.spectral_centroid(y=seg, sr=SR, n_fft=2048, hop_length=512)[0]
    cs = float(np.median(cent))
    return med, sd, cs, rms


def build(source, prime_start, out_path, max_segs=40, rms_target=0.11,
          med_min=100, med_max=140, cent_max=1900, rms_min=0.05, rms_max=0.22,
          sep_db=24, silent=0.15):
    y, _ = librosa.load(source, sr=SR, mono=True)
    y = y.astype(np.float32)
    p0, p1 = int(prime_start * SR), int((prime_start + W) * SR)
    prime, _ = librosa.effects.trim(y[p0:p1].astype(np.float32), top_db=18)
    prime = prime.astype(np.float32)
    print(f"prime {prime_start:.1f}-{prime_start+W:.1f}s -> {len(prime)/SR:.1f}s "
          f"medF0={analyze(y, prime_start, prime_start+W)[0]:.0f}Hz "
          f"std={analyze(y, prime_start, prime_start+W)[1]:.0f}")

    runs = librosa.effects.split(y, top_db=sep_db, frame_length=1024, hop_length=HOP)
    cand = []
    for (a, b) in runs:
        if b - a < int(1.0 * SR):
            continue
        if a < p1 and b > p0:
            continue
        s, _ = librosa.effects.trim(y[a:b], top_db=16)
        if len(s) < int(1.0 * SR):
            continue
        r = float(np.sqrt(np.mean(s**2)))
        if not (rms_min <= r <= rms_max):
            continue
        med, sd, cs, _ = analyze(y, a / SR, b / SR)
        if med < med_min or med > med_max:
            continue
        if cs > cent_max:
            continue
        calm = -med - 1.0 * sd - cs / 40 - 4.0 * abs(r - rms_target)
        cand.append((calm, r, len(s) / SR, med, sd, cs, a / SR, b / SR, s.astype(np.float32)))
    cand.sort(reverse=True)
    cand = cand[:max_segs]
    total = sum(x[2] for x in cand)
    print(f"using {len(cand)} segs, {total:.1f}s (medF0 {min(x[3] for x in cand):.0f}-"
          f"{max(x[3] for x in cand):.0f}Hz, cent max {max(x[5] for x in cand):.0f})")

    sil = np.zeros(int(silent * SR), dtype=np.float32)
    out = prime
    for calm, r, d, med, sd, cs, a, b, s in cand:
        out = np.concatenate([out, sil, s * (0.2 / r)])
    out = (out * (0.2 / (np.sqrt(np.mean(out**2)) + 1e-12))).astype(np.float32)
    sf.write(out_path, out, SR)
    print(f"ref: {len(out)/SR:.1f}s -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prime", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-segs", type=int, default=40)
    ap.add_argument("--med-min", type=int, default=100)
    ap.add_argument("--med-max", type=int, default=140)
    ap.add_argument("--cent-max", type=int, default=1900)
    ap.add_argument("--rms-target", type=float, default=0.11)
    ap.add_argument("--source", default="tate_voice_to_clone.wav")
    args = ap.parse_args()
    base = __import__("os").path.dirname(__file__)
    build(__import__("os").path.join(base, args.source), args.prime, args.out,
          max_segs=args.max_segs, med_min=args.med_min, med_max=args.med_max,
          cent_max=args.cent_max, rms_target=args.rms_target)