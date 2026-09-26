"""Selects which registered voice narrates this run.

Round-robin selection cycles through the enabled voices so each gets a turn
(trump -> arnold -> trump -> arnold; a newly enabled voice joins the cycle).
"""

from src.agents.voice_cast.voices import get_enabled_voices, get_voice
from src.agents.voice_cast.writing_styles import get_style

# Round-robin cursor: modulo by the number of enabled voices.
_round_robin_index = 0


def pick_voice(preferred: str = "") -> tuple:
    """Pick a voice id + entry for this video by strict round-robin.

    preferred forces a voice id if it exists and is enabled. With two enabled
    voices this alternates A, B, A, B, and a newly enabled voice joins the
    cycle automatically."""
    global _round_robin_index
    if preferred:
        voice = get_voice(preferred)
        if voice and voice.get("enabled"):
            return preferred, voice
        print(f"  [voice] Unknown or disabled voice '{preferred}' - picking automatically")

    candidates = get_enabled_voices()
    voice_id, voice_cfg = candidates[_round_robin_index % len(candidates)]
    _round_robin_index += 1
    return voice_id, voice_cfg


def get_writing_style(voice_id: str, voice: dict) -> dict:
    """Resolve the writing style dict for a voice."""
    return get_style(voice.get("writing_style"))
