#!/usr/bin/env python3
"""Test publishing an existing rendered video to YouTube.

Usage:
    python test_youtube_upload.py                          # uses the latest video in output/
    python test_youtube_upload.py <video_path> "<topic>"   # custom video + topic

Requirements:
    - client_secrets.json (OAuth Desktop client) in project root
    - YouTube Data API v3 enabled for the Google account

The video is uploaded as 'private' by default. Change YOUTUBE_PRIVACY_STATUS
in .env to 'public' or 'unlisted' to change that.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import YOUTUBE_PRIVACY_STATUS
from src.services.youtube_upload import generate_metadata, publish_video


def _find_latest_video() -> str:
    videos = sorted(
        glob.glob(os.path.join("output", "*.mp4")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not videos:
        raise FileNotFoundError("No .mp4 found in output/. Pass a video path instead.")
    return videos[0]


def _load_last_script(topic: str) -> dict:
    """Load the last generated script (raw_json_response.txt) if its topic matches."""
    script = {"topic": topic, "lines": []}
    if os.path.exists("raw_json_response.txt"):
        try:
            saved = json.load(open("raw_json_response.txt"))
            if saved.get("topic") and saved.get("topic").lower() == topic.lower():
                script = saved
                print(f"   Reusing script: {len(script['lines'])} lines from raw_json_response.txt")
        except (json.JSONDecodeError, OSError):
            pass
    return script


def main():
    parser = argparse.ArgumentParser(description="Test YouTube publishing.")
    parser.add_argument("video", nargs="?", default=None, help="Path to the video file")
    parser.add_argument("topic", nargs="?", default=None, help="Video topic (used for metadata)")
    args = parser.parse_args()

    video_path = args.video or _find_latest_video()

    if args.topic:
        topic = args.topic
    else:
        topic = os.path.splitext(os.path.basename(video_path))[0].replace("_", " ")
        topic = topic.rstrip(".")

    script = _load_last_script(topic)

    print("Testing YouTube Publishing\n" + "=" * 60)
    print(f"  🎬 Video:  {video_path}")
    print(f"  📝 Topic:  {topic}")
    print(f"  🔒 Privacy: {YOUTUBE_PRIVACY_STATUS}")

    if not os.path.exists("client_secrets.json"):
        print("\n❌ client_secrets.json not found in project root.")
        print("   Enable the YouTube Data API v3 in Google Cloud Console and")
        print("   download your OAuth Desktop client credentials.")
        sys.exit(1)

    print("\n1️⃣  Generating title & description...")
    meta = generate_metadata(topic, script)
    print(f"   Title: {meta['title']}")
    print(f"   Description ({len(meta['description'].split())} words):")
    print(f"   {meta['description'][:200]}...")
    print(f"   Tags ({len(meta['tags'])}): {', '.join(meta['tags'][:6])}")

    print("\n2️⃣  Publishing...")
    video_id = publish_video(video_path, topic, script)

    print("\n" + "=" * 60)
    print(f"✅ Success! Video published:")
    print(f"   https://youtu.be/{video_id}")


if __name__ == "__main__":
    main()