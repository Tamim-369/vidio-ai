"""Entry point: `python src/main.py`.

    uv run src/main.py --batch=10 --no-upload

One character, one quote, one video. The only question the CLI asks is how many.
"""
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
        description="Anti-wisdom quote videos: one character, one quote, "
                    "one video.")
    parser.add_argument("--batch", type=int, default=0, metavar="N",
                        help="Make N videos in one run (default 1)")
    parser.add_argument("--voice", default="",
                        help="Force one voice id for every video instead of "
                             "rotating through the enabled ones")
    parser.add_argument("--script-only", action="store_true",
                        help="Only write the quote — no audio, video or upload")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to YouTube after rendering "
                             "(overrides AUTO_PUBLISH)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Keep the video local (overrides AUTO_PUBLISH)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Flags win over the AUTO_PUBLISH env default, which defaults to off.
    publish = False if args.no_upload else (True if args.upload else AUTO_PUBLISH)

    if args.batch:
        run_batch(count=args.batch, publish=publish, voice=args.voice,
                  script_only=args.script_only)
    else:
        create_video(character=args.voice, publish=publish,
                     script_only=args.script_only)


if __name__ == "__main__":
    main()
