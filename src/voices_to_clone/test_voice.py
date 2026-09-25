#!/usr/bin/env python3
"""Test a registered voice end-to-end before enabling it in the video generator.

Usage:
    source ../../.venv/bin/activate
    python test_voice.py --voice donald-trump "text to speak"
    python test_voice.py                              # list voices + interactive probe
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Make project root importable regardless of cwd (this script lives in src/voices_to_clone/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.voices import get_all_voices, get_voice
from src.config.writing_styles import get_style
from src.services.tts import generate_audio


def list_voices():
    print("Registered voices:")
    for vid, v in get_all_voices().items():
        status = "enabled" if v.get("enabled") else "disabled"
        print(f"  - {vid:16} {v['name']:28} [{v['engine']}] [{v.get('writing_style')}] ({status})")
    print("\nWriting styles:")
    for sid in ["narrator", "trump", "andrew_tate", "arnold"]:
        style = get_style(sid)
        print(f"  - {sid:16} {style['name']}")


def _slugify(text: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in text.lower())[:30].rstrip("_")
    return slug or "voice"


def speak(voice_id: str, text: str, out_name: str = ""):
    voice = get_voice(voice_id)
    if not voice:
        raise SystemExit(f"Unknown voice '{voice_id}'. Use --list-voices to see available voices.")

    style = get_style(voice.get("writing_style"))
    print(f"Voice: {voice['name']} ({voice['engine']}) — writing style: {style['name']}")

    lines = [{"id": 1, "text": text}]
    lines = generate_audio(lines, voice=voice)

    # Keep each test in its own file so A/B dial-tuning doesn't overwrite results.
    stamp = out_name or f"{voice_id}_{_slugify(text)}"
    keep = Path("voice_tests") / f"{stamp}.wav"
    keep.parent.mkdir(exist_ok=True)
    shutil.copy2(lines[0]["audio_path"], keep)
    print(f"\nSaved: {keep}  ({lines[0]['actual_duration']:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Test a voice from the registry")
    parser.add_argument("--voice", default="", help="Voice id from the registry")
    parser.add_argument("text", nargs="*", help="Text for the voice to speak")
    parser.add_argument("--out", default="", help="Output filename base (saved under voice_tests/)")
    parser.add_argument("--list-voices", action="store_true", help="List voices and styles, then exit")
    args = parser.parse_args()

    if args.list_voices:
        list_voices()
        return

    if not args.voice:
        parser.print_help()
        print("\nAvailable voices:")
        list_voices()
        return

    text = " ".join(args.text) or input("Enter text to speak: ").strip()
    if not text:
        raise SystemExit("No text given.")
    speak(args.voice, text, out_name=args.out)


if __name__ == "__main__":
    main()