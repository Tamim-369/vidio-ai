"""Voice manager — selects voices for the video pipeline.

Responsibilities:
- Round-robin selection: cycle through enabled voices so each gets a turn
  (trump → arnold → trump → arnold; a new enabled voice joins the cycle).
- Resolve a voice's writing style for script generation.
"""

from src.config.voices import get_all_voices, get_enabled_voices
from src.config.writing_styles import get_style

# Round-robin cursor: modulo by the number of enabled voices.
_round_robin_index = 0


def list_voices() -> None:
    """Print every voice in the registry with its status."""
    for vid, v in get_all_voices().items():
        status = "enabled" if v.get("enabled") else "disabled"
        print(f"  - {vid:16} {v['name']:28} [{v['engine']}] [{v.get('writing_style')}] ({status})")


def pick_voice(preferred: str = "", exclude: set = None) -> tuple:
    """Pick a voice id + entry for this video by strict round-robin.

    preferred: force a specific voice id if it exists and is enabled.
    exclude:   set of voice ids to skip this turn (used for batch variety).

    With two enabled voices this alternates A, B, A, B, ... across a batch.
    When a new voice is enabled it is inserted into the cycle automatically.
    """
    global _round_robin_index
    exclude = exclude or set()
    if preferred and preferred not in exclude:
        voice = get_voice(preferred)
        if voice and voice.get("enabled"):
            return preferred, voice
        print(f"  [voice] Unknown or disabled voice '{preferred}' — picking automatically")

    candidates = [(vid, v) for vid, v in get_enabled_voices() if vid not in exclude]
    if not candidates:
        candidates = get_enabled_voices()

    voice_id, voice_cfg = candidates[_round_robin_index % len(candidates)]
    _round_robin_index += 1
    return voice_id, voice_cfg


def get_writing_style(voice_id: str, voice: dict) -> dict:
    """Resolve the writing style dict for a voice."""
    return get_style(voice.get("writing_style"))