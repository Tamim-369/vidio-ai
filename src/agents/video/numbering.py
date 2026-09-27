"""The per-video number in the channel's titles and descriptions.

Titles carry ``| Wordz Of Wizdom #1293`` so a video is findable by its number and
the channel reads as a numbered library rather than an undated feed. That only
works if a number is never issued twice, so the counter is persisted.

The number advances when a video *completes*, not when it is published, which
means a `--no-upload` preview consumes numbers. That is the deliberate trade: a
gap in the numbering is invisible to a viewer, whereas two videos sharing a
number is visible, and a preview that shows the real schedule is worth more than
conservering a counter.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Absolute for the same reason as the quote pool's STATE_FILE: a relative path
# resolves against the CWD, so running from anywhere but the repo root would
# restart the count and reissue numbers already on the channel.
_REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_FILE = os.getenv("VIDEO_NUMBER_FILE", str(_REPO_ROOT / "video_number.json"))

FIRST_NUMBER = 1


def _load(path: str = None) -> int:
    try:
        with open(path or STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return FIRST_NUMBER
    if isinstance(data, dict):
        nxt = data.get("next")
        if isinstance(nxt, int) and nxt >= FIRST_NUMBER:
            return nxt
    return FIRST_NUMBER


def _save(next_number: int, path: str = None) -> None:
    target = path or STATE_FILE
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump({"next": next_number}, f, indent=2)


def peek(path: str = None) -> int:
    """The number the next completed video will carry. Does not consume it."""
    return _load(path)


def advance(path: str = None) -> int:
    """Consume the current number and return the one after it."""
    nxt = _load(path) + 1
    _save(nxt, path)
    return nxt


def reset(path: str = None) -> None:
    """Start counting again from the first video."""
    _save(FIRST_NUMBER, path)
