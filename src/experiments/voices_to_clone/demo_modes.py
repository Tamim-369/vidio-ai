#!/usr/bin/env python3
"""Speak hand-written scripts - each in its voice's OWN native mode - and compare."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

VOICE_SCRIPTS = {
    "donald-trump": (
        "Let me tell you, this is the greatest economy we have ever built. " 
        "Tremendous. Really incredible. And the fake news won't tell you that. " 
        "They said we couldn't do it. They said it would be a complete disaster. " 
        "...and you know what? We did it anyway. Nobody does it like this. Nobody."
    ),
    "arnold-schwarzenegger": (
        "Listen to me. You have to understand this. Success is not a gift. " 
        "It is a weight. You load the bar. You squat. You fail. You get up and squat again. " 
        "That is the formula. There is no shortcut, there is no magic train. " 
        "...and if you want it bad enough, NOTHING is going to stop you. That is everything."
    ),
}

STYLE = """Style rules for this script:
- Short, punchy lines. One idea per line.
- The voice's natural energy: explosive at openers, drops to a quiet '...' beat for emphasis, then build to the caps moment.
- ONE genuine caps moment per script (severe emphasis).
- No hyphens as pauses. No double quotes. Do NOT include any stage directions."""


def run(voice_id: str, text: str):
    print(f"\n{'='*60}\n{voice_id}\n{'='*60}", flush=True)
    subprocess.run(
        [sys.executable, "src/experiments/voices_to_clone/test_voice.py", "--voice", voice_id, "--out", f"mode_{voice_id}", text],
        check=True,
    )


if __name__ == "__main__":
    for vid, script in VOICE_SCRIPTS.items():
        run(vid, script)
    print("\nDone: src/experiments/voice_tests/mode_donald-trump.wav, mode_arnold-schwarzenegger.wav")