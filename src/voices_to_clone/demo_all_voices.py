#!/usr/bin/env python3
"""Speak one hand-written script with all three cloned voices and compare."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCRIPT = (
    "The market just crashed again. Twenty percent gone in one week. "
    "And you know what the news is telling you? That it's over. That the good times are done. "
    "...but that's exactly what they said in 2008, in 2020, in every single crash before this one. "
    "The people who sold that day made a fortune for the banks who bought it. "
    "The people who STAYED and bought the panic are the ones you see driving the Lamborghinis. "
    "So here's my question to you: are you the seller, or are you the buyer?"
)


def run(voice_id: str):
    keep = str(Path("voice_tests") / f"demo_{voice_id}.wav")
    print(f"\n{'='*60}\n{voice_id}\n{'='*60}", flush=True)
    subprocess.run(
        [sys.executable, "src/voices_to_clone/test_voice.py", "--voice", voice_id, "--out", f"demo_{voice_id}", SCRIPT],
        check=True,
    )
    print(f"-> {keep}")


if __name__ == "__main__":
    for vid in ["donald-trump", "arnold-schwarzenegger"]:
        run(vid)
    print("\nBoth done: demo_donald-trump.wav, demo_arnold-schwarzenegger.wav")