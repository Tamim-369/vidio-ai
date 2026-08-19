#!/usr/bin/env python3
"""Generate 20 unique video-ready topics for the faceless YouTube pipeline.

Pulls from Reddit JSON, Wikipedia categories, and RSS feeds (no auth, no AI),
filters + scores heuristically, dedupes against used_topics.json, and prints
the top 20 topics as JSON. A copy is saved to topics/batch_<timestamp>.json.

Usage:
    python generate_topics.py                 # default: limit=100 per source
    python generate_topics.py --limit 10      # smaller pulls for quick testing
    python generate_topics.py --target 20     # number of topics to output
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.topic_generator import run_topic_generation


def main():
    parser = argparse.ArgumentParser(description="Generate unique video topics.")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max posts per Reddit subreddit (default 100)")
    parser.add_argument("--target", type=int, default=20,
                        help="Number of topics to output (default 20)")
    parser.add_argument("--save-only", action="store_true",
                        help="Only save to topics/, don't print the full JSON")
    parser.add_argument("--no-browser", action="store_true",
                        help="Use requests + cookie instead of headless Chromium")
    args = parser.parse_args()

    topics = run_topic_generation(limit=args.limit, target=args.target, use_browser=not args.no_browser)

    print(f"\n{'=' * 60}")
    print(f"Top {len(topics)} topics:")
    if not args.save_only:
        print(json.dumps(topics, indent=2))
    else:
        for i, t in enumerate(topics, 1):
            print(f"{i:2d}. [{t['category']:9}] {t['title']}")


if __name__ == "__main__":
    main()