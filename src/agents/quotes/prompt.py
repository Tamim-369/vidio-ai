"""One prompt per character.

A character's voice and its jokes have to agree, so the prompt is a property of
the character rather than of the quote agent. Each module here exposes:

    build_prompt(subject, n) -> str   the prompt, with its variables filled
    MIN_CHARS / MAX_CHARS             the length window that prompt asks for

Character-specific length bounds live with the prompt because they are part of
its specification, not a global setting: these prompts cap quotes at 100
characters, so filtering at 200 for them would contradict what they were told.

Every character prompt ignores ``subject``: the quotes are about the character,
not about a per-video topic. The signature is kept uniform so one interface
covers every module. The subject is still used, in assemble.py, to label the
video's title and filename.

A character with no module of its own falls back to ``shared``, which still
takes a subject because a generic prompt has no persona to fall back on. The
fallback is deliberately visible in the agent's log line so a missing prompt is
obvious rather than silent.
"""
from __future__ import annotations

from src.agents.quotes import prompt_shared as shared

# voice id -> prompt module
_PROMPTS = {
    "donald-trump": "don_tzu",
    "andrew-tate": "andru_tatte",
    "arnold-schwarzenegger": "brolexander",
}


def _load(name: str):
    if name == "shared":
        return shared
    from importlib import import_module
    return import_module(f"src.agents.quotes.prompt_{name}")


def get_prompt(character: str = ""):
    """Return the prompt module for a character, or the shared fallback.

    Args:
        character: the voice id, e.g. "donald-trump". Unknown and empty ids get
            the shared prompt so a new character cannot break generation.
    """
    name = _PROMPTS.get(character, "shared")
    return _load(name)


def characters_with_prompts() -> list:
    """Voice ids that have a prompt of their own, in registration order."""
    return list(_PROMPTS)


__all__ = ["get_prompt", "characters_with_prompts", "shared"]
