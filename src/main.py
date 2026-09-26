"""Entry point: `python src/main.py`."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.video import create_video, run_batch

# AUTO_PUBLISH=1 in .env publishes after rendering; --upload/--no-upload win.
AUTO_PUBLISH = os.getenv("AUTO_PUBLISH", "0") == "1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Anti-wisdom quote video (Groq quotes -> voice -> video)")
    parser.add_argument("--batch", type=int, default=0, metavar="N",
                        help="Render N videos in one run")
    parser.add_argument("--quotes", type=int, default=2, metavar="N",
                        help="Quotes per video (1-2; the prompt's limit is 2)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to YouTube after rendering (overrides AUTO_PUBLISH)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Render WITHOUT uploading to YouTube (overrides AUTO_PUBLISH)")
    parser.add_argument("--voice", default="",
                        help="Force a specific voice id")
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

    if args.batch:
        run_batch(count=args.batch, publish=publish, voice=args.voice,
                  script_only=args.script_only)
    else:
        create_video(n_quotes=args.quotes, publish=publish, voice=args.voice,
                     script_only=args.script_only)


if __name__ == "__main__":
    main()
