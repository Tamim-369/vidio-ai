"""Resolves a voice id to the writing style its quotes should be written in.

Which character narrates a video is not decided here. It is a function of the
video's number in the channel: the planner puts video N on voice
``(N - 1) % len(enabled)``, so #1 is the first enabled voice, #2 the second, and
a batch continues the cycle across runs because the number is persisted. There is
no cursor to keep and no state to fall out of step with the plan -- the number
*is* the state, and deriving the voice from it cannot disagree with itself the
way a separate cursor could.
"""

from src.agents.voice_cast.writing_styles import get_style


def get_writing_style(voice_id: str, voice: dict) -> dict:
    """Resolve the writing style dict for a voice."""
    return get_style(voice.get("writing_style"))
