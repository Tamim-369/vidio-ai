"""CLI entry point for the faceless YouTube video pipeline.

Parses command-line arguments and dispatches to the orchestration functions
in src/pipeline/. All pipeline logic lives there; this file only wires the
provided flags to the right flow (single topic, batch, or voice listing).
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Bootstrap environment before importing pipeline modules: .env values are read
# at import time by every module that owns a knob, so the file must be on disk
# before the `from src...` lines below.
load_dotenv()

# Keep HuggingFace model downloads inside the project when HF_HOME is set.
hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline import create_video, run_batch
from src.agents.voice import agent as voice_manager


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Faceless YouTube video pipeline")
    parser.add_argument("topic", nargs="?", default=None, help="Single topic to make a video for")
    parser.add_argument("--batch", action="store_true", help="Generate topics and make videos for all")
    parser.add_argument("--use-saved", action="store_true", help="Use saved topic batch instead of generating")
    parser.add_argument("--upload", action="store_true", help="Upload to YouTube after rendering (default)")
    parser.add_argument("--no-upload", action="store_true", help="Render WITHOUT uploading to YouTube")
    parser.add_argument("--voice", default="", help="Force a specific voice id (see --list-voices)")
    parser.add_argument("--list-voices", action="store_true", help="List all registered voices and exit")
    parser.add_argument("--script-only", action="store_true",
                        help="Only generate topics + scripts (no assets, audio, video, or upload)")
    parser.add_argument("--limit", type=int, default=100, help="Posts per source when researching topics")
    parser.add_argument("--target", type=int, default=24, help="How many topics to research")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    publish = not args.no_upload

    if args.list_voices:
        voice_manager.list_voices()
    elif args.batch:
        run_batch(generate=not args.use_saved, limit=args.limit, target=args.target, publish=publish,
                  voice=args.voice, script_only=args.script_only)
    elif args.topic:
        create_video(args.topic, publish=publish, voice=args.voice, script_only=args.script_only)
    else:
        # Default: generate topics from Reddit and make videos for all of them
        run_batch(generate=True, limit=args.limit, target=args.target, publish=publish,
                  voice=args.voice, script_only=args.script_only)


if __name__ == "__main__":
    main()