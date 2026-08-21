#!/usr/bin/env python3
"""Standalone content-research pipeline entry point.

Sources fresh ideas per niche (ancient mysteries / WW2 secret experiments /
obscure history) via Wikipedia + Ollama (minimax-m3), dedups against
everything already produced, scores, and writes the approved queue to
topics/batch_<timestamp>.json. Review the printed list before producing —
it's the human gate between research and production.

Usage:
    python research_topics.py                 # default: 12 topics
    python research_topics.py --target 20     # more topics
    python research_topics.py --no-miner      # skip channel mining
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.research_pipeline import run_research_pipeline


def main():
    parser = argparse.ArgumentParser(description="Research new video topics.")
    parser.add_argument("--target", type=int, default=24,
                        help="Number of topics to output (default 24)")
    parser.add_argument("--no-miner", action="store_true",
                        help="Skip competitor-channel mining (default: include)")
    args = parser.parse_args()

    if args.no_miner:
        # Temporarily disable the miner by monkeypatching the import target.
        import src.services.research_pipeline as rp
        rp.fetch_channel_candidates = lambda **kw: []
        print("Channel mining disabled (--no-miner)")

    run_research_pipeline(target=args.target)


if __name__ == "__main__":
    main()