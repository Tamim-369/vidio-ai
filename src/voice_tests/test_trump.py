#!/usr/bin/env python3
"""Render a Donald Trump test video from the cached script.

First run generates a fresh test script (research + LLM + images) and saves it
under src/voice_tests/scripts + src/voice_tests/assets. Later runs reuse the
cache, so only TTS + assembly run — no research/LLM/downloads.

Usage (from project root, with .venv active):
    python src/voice_tests/test_trump.py                      # render cached script
    python src/voice_tests/test_trump.py --topic "..."        # regenerate script for a topic, then render
    python src/voice_tests/test_trump.py --topic "..." --no-render   # only generate script
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.voice_tests.voice_test_utils import main

if __name__ == "__main__":
    main("donald-trump", "A historical military failure with exact numbers, distances and years")