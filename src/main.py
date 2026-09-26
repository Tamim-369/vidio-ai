"""CLI entry point for the anti-wisdom quote video pipeline.

Parses command-line arguments and dispatches to the orchestration functions in
src/pipeline/. All pipeline logic lives there; this file only wires the provided
flags to the right flow (one video, a batch of them, or voice listing).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import AUTO_PUBLISH
from src.pipeline.batch import run_batch
from src.pipeline.quote_video import create_video
from src.services import voice_manager


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Anti-wisdom quote video pipeline (Groq quotes -> voice -> video)")
    parser.add_argument("--batch", type=int, default=0, metavar="N",
                        help="Render N videos in one run")
    parser.add_argument("--quotes", type=int, default=2, metavar="N",
                        help="Quotes per video (1-2; the prompt's limit is 2)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to YouTube after rendering (overrides AUTO_PUBLISH)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Render WITHOUT uploading to YouTube (overrides AUTO_PUBLISH)")
    parser.add_argument("--voice", default="",
                        help="Force a specific voice id (see --list-voices)")
    parser.add_argument("--list-voices", action="store_true",
                        help="List all registered voices and exit")
    parser.add_argument("--script-only", action="store_true",
                        help="Only generate the quotes (no audio, video, or upload)")
    return parser


def _resolve_publish(args: argparse.Namespace) -> bool:
    """Flags win over the AUTO_PUBLISH env default, which defaults to off."""
    if args.no_upload:
        return False
    if args.upload:
        return True
    return AUTO_PUBLISH


def main() -> None:
    args = build_parser().parse_args()
    publish = _resolve_publish(args)

    if args.list_voices:
        voice_manager.list_voices()
    elif args.batch:
        run_batch(count=args.batch, publish=publish, voice=args.voice,
                  script_only=args.script_only)
    else:
        # Default: one video.
        create_video(n_quotes=args.quotes, publish=publish, voice=args.voice,
                     script_only=args.script_only)


if __name__ == "__main__":
    main()
