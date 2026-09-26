"""Picks which registered voice narrates this run."""
from src.agents.voice_cast.agent import get_writing_style, pick_voice
from src.agents.voice_cast.voices import (
    VOICES,
    get_all_voices,
    get_enabled_voices,
    get_voice,
    pick_quote_author,
)
from src.agents.voice_cast.writing_styles import WRITING_STYLES, get_style

__all__ = [
    "VOICES",
    "WRITING_STYLES",
    "get_all_voices",
    "get_enabled_voices",
    "get_style",
    "get_voice",
    "get_writing_style",
    "pick_quote_author",
    "pick_voice",
]
