#!/usr/bin/env python3
"""Render a Donald Trump test video from the cached quotes.

First run fetches fresh quotes (Groq quote agent) and saves them under
src/experiments/voice_tests/scripts + src/experiments/voice_tests/assets. Later runs reuse the cache,
so only TTS + assembly run — no Groq calls, no downloads.

Usage (from project root, with .venv active):
    python src/experiments/voice_tests/test_trump.py                          # render cached quotes
    python src/experiments/voice_tests/test_trump.py --quotes 3               # fetch 3 fresh quotes, then render
    python src/experiments/voice_tests/test_trump.py --quotes 3 --no-render   # only generate quotes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.experiments.voice_tests.voice_test_utils import main

if __name__ == "__main__":
    main("donald-trump", 2)
