"""Selects which registered voice narrates this run.

Rotation is strict and persisted: each video takes the next enabled voice, so
consecutive videos are narrated by different characters. Because every voice
declares its own ``subject``, rotating the voice also rotates the subject -- one
video is War with Don Tzu, the next is Weightlifting with Brolexander, and so on.

The cursor lives in a file rather than a module global because a batch runs in
one process while separate ``python src/main.py`` invocations do not. An
in-memory counter would restart at zero on every new run, so every invocation
would open on the same character.
"""

import json
import os

from src.agents.voice_cast.voices import get_enabled_voices, get_voice
from src.agents.voice_cast.writing_styles import get_style

# Inside src/ so cleanup_temp() cannot wipe it. Override with
# VOICE_ROTATION_FILE.
ROTATION_FILE = os.getenv("VOICE_ROTATION_FILE", "src/state/voice_rotation.json")


def _load_last_voice(path: str = None) -> str:
    """The voice id used by the previous video, or "" if unknown."""
    try:
        with open(path or ROTATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return ""
    return data.get("last_voice", "") if isinstance(data, dict) else ""


def _save_last_voice(voice_id: str, path: str = None) -> None:
    path = path or ROTATION_FILE
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"last_voice": voice_id}, f, indent=2)


def pick_voice(preferred: str = "", path: str = None) -> tuple:
    """Pick a voice id + entry for this video, advancing the rotation.

    preferred forces a voice id if it exists and is enabled, and still advances
    the cursor so the run after an override continues the cycle instead of
    repeating the overridden voice.
    """
    candidates = get_enabled_voices()
    if not candidates:
        raise RuntimeError("no enabled voices to narrate with")

    if preferred:
        voice = get_voice(preferred)
        if voice and voice.get("enabled"):
            _save_last_voice(preferred, path)
            return preferred, voice
        print(f"  [voice] Unknown or disabled voice '{preferred}' - picking automatically")

    ids = [vid for vid, _ in candidates]
    last = _load_last_voice(path)
    # Start after the previous pick. An unknown or since-disabled last voice
    # falls back to the front of the cycle rather than skipping a turn.
    index = (ids.index(last) + 1) % len(ids) if last in ids else 0
    voice_id = ids[index]
    _save_last_voice(voice_id, path)
    # candidates hold (id, voice) pairs, so index the pair out rather than
    # returning the tuple itself.
    return voice_id, candidates[index][1]


def get_writing_style(voice_id: str, voice: dict) -> dict:
    """Resolve the writing style dict for a voice."""
    return get_style(voice.get("writing_style"))
