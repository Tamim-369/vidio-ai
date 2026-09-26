#!/usr/bin/env python3
"""Simulate production voiceover rendering for the cloned voices.

Reproduces the exact production condition: the same script line shape
(id/text/is_first/loud), the same Chatterbox model, the same per-line DSP
chain, and the same parallel-worker renderer (`_generate_chatterbox`) that
the main pipeline calls. Only the audio directory is redirected and the line
set is custom/randomized so a full batch never has to run.

Usage:
    uv run python src/experiments/voice_tests/sim_production.py [--voices donald-trump arnold-schwarzenegger] [--seed N] [--workers 2]

Output lands in src/experiments/voice_tests/audio/sim_production/<voice_id>/<line_id>.wav
plus a results.json with per-line validation (duration, peak, RMS, longest
silent run) and a pass/fail summary printed to the console.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import numpy as np
import soundfile as sf

SIM_DIR = Path(__file__).resolve().parent / "audio" / "sim_production"

# Realistic story-lab narration - the style qwen2.5:3b actually produces:
# hard figures, dates, em-dashes, punchy short lines, no fluffy adjectives.
_POOL = [
    "The raid came at dawn and it cost the enemy one hundred and twelve troops in under six minutes.",
    "By 1944 the fleet had lost four hundred ships, and the admiral still refused to change course.",
    "Here is the number you do not hear: seventeen thousand dollars was paid for every single ton.",
    "The border crossing stayed open for forty-nine days, and two hundred thousand people moved through it in that time.",
    "That is the whole problem, folks, in one sentence: the plan was never real.",
    "In 1969 the review board demanded the files, and every single page came back blank.",
    "She inherited the office at thirty-one, ran it for twelve years, and left with profit at one point two billion.",
    "The treaty was signed, the ink barely dried, and then the army marched anyway.",
    "Look at the map: six hundred miles of it, and not one checkpoint actually works.",
    "The engineers found the fault in 1987, buried it in 1988, and we paid for it in 2023.",
    "You have heard the easy version. Here is the one the transcript leaves out.",
    "The reactor went critical, the crew had ninety seconds, and the log went silent at oh four hundred.",
    "Three billion went to the contractor, the bridge opened late, and the newsletter called it a success.",
    "The verdict came down in forty minutes. The appeal was filed in L.A. and it took four years.",
    "Nobody says this part out loud, but the subsidy is the reason the price never fell.",
    "By the time the team arrived, the site had been stripped, and the paperwork showed a full inventory.",
    "The audit found two million missing and the boss answered: account error, no further comment.",
    "One patrol, fifteen soldiers, and the platoon came back with eighty cartons of unopened supplies.",
    "The number that killed the deal was eight percent, because eight percent is larger than the law allows.",
    "He called it a withdrawal, but the tanks moved up, and the guard count doubled overnight.",
]

# Key -> heading for the validation summary.
_METRICS = ("duration_s", "peak", "rms", "longest_silence_ms")


def _make_lines(rng: random.Random, voice_id: str, n: int) -> list:
    """Randomize a realistic line set (production dict shape)."""
    lines = []
    for i in range(n):
        text = rng.choice(_POOL)
        while lines and lines[-1]["text"] == text:  # no adjacent dupes in the sim
            text = rng.choice(_POOL)
        lines.append({
            "id": f"{voice_id.replace('-', '')}_sim_{i + 1}",
            "text": text,
            "is_first": i == 0,
            "loud": rng.random() < 0.2,
        })
    # At least one loud line so the loud path (LOUD_GAIN + de_shout) is exercised.
    lines[rng.randrange(n)]["loud"] = True
    return lines


def _max_silent_ms(wav: np.ndarray, sr: int) -> float:
    """Longest run of near-zero samples (the 'speech breaking' check)."""
    quiet = np.abs(wav) < 0.01 * max(abs(float(wav.max(initial=0))), 1e-6)
    best = cur = 0
    for q in quiet:
        cur = cur + 1 if q else 0
        best = max(best, cur)
    return best / sr * 1000.0


def _validate(wav: np.ndarray, sr: int) -> dict:
    return {
        "duration_s": round(len(wav) / sr, 3),
        "peak": round(float(np.max(np.abs(wav), initial=0.0)), 4),
        "rms": round(float(np.sqrt(np.mean(wav ** 2) + 1e-12)), 4),
        "longest_silence_ms": round(_max_silent_ms(wav, sr), 1),
    }


def _reuse_or_render(lines, voice, voice_dir):
    """Respire previously rendered lines (no model re-run) -> (to_render, reused)."""
    from src.agents.voiceover import _generate_chatterbox
    to_render, reused = [], []
    for ln in lines:
        wav_path = voice_dir / f"{ln['id']}.wav"
        if wav_path.exists():
            try:
                data, sr = sf.read(str(wav_path))
                if len(data) > int(sr * 0.2) and float(np.max(np.abs(data), initial=0)) > 0.05:
                    ln["audio_path"] = str(wav_path)
                    ln["actual_duration"] = len(data) / sr
                    reused.append(ln)
                    continue
            except Exception:
                pass
        to_render.append(ln)
    if to_render:
        _generate_chatterbox(to_render, voice, str(voice_dir))
    return to_render + reused


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="+",
                    default=["donald-trump", "arnold-schwarzenegger"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lines", type=int, default=3, help="random lines per voice")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Apply the worker cap BEFORE importing the pipeline so the env var the
    # settings module snapshots at import time matches what the user asked for.
    os.environ["TTS_WORKERS"] = str(args.workers)
    from src.agents.voice_cast import get_voice
    from src.agents.voiceover import _generate_chatterbox

    SIM_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for voice_id in args.voices:
        voice = get_voice(voice_id)
        if not voice:
            print(f"!! unknown voice: {voice_id}")
            continue
        if voice.get("engine") != "chatterbox" or not voice.get("ref_audio"):
            print(f"!! {voice_id}: not a chatterbox clone, skipping")
            continue
        lines = _make_lines(rng, voice_id, args.lines)
        voice_dir = SIM_DIR / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {voice['name']} ({voice_id}) - {len(lines)} lines ===",
              flush=True)
        for ln in lines:
            print(f"    [{ln['id']}]{' [LOUD]' if ln['loud'] else ''} {ln['text']}",
                  flush=True)

        _reuse_or_render(lines, voice, voice_dir)
        line_res = []
        all_ok = True
        for ln in lines:
            path = str(ln["audio_path"])
            wav, sr = sf.read(str(path))
            m = _validate(wav, sr)
            ok = (np.isfinite(wav).all()
                  and m["duration_s"] > 0.2
                  and m["peak"] > 0.05
                  and 0.005 < m["rms"] < 1.0
                  and m["longest_silence_ms"] < 1000.0)
            all_ok &= bool(ok)
            line_res.append({
                "id": ln["id"], "text": ln["text"], "loud": ln["loud"],
                "dur_from_pipeline": round(ln["actual_duration"], 3),
                "valid": ok, "metrics": m,
            })
            print(f"    {'OK ' if ok else 'FAIL'} {os.path.basename(path)} "
                  f"{m['duration_s']}s peak={m['peak']} rms={m['rms']} "
                  f"silence={m['longest_silence_ms']}ms", flush=True)
        results[voice_id] = {"voice": voice["name"], "lines": line_res, "all_ok": all_ok}

    with open(SIM_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    for vid, r in results.items():
        print(f"{r['voice']}: {'PASS' if r['all_ok'] else 'FAIL'} ({len(r['lines'])} lines)")
    print(f"\nwavs + results.json written to {SIM_DIR}")


if __name__ == "__main__":
    main()