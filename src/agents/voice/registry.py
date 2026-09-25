"""Voice registry — data-driven list of voices available to the video generator.

The main branch pipeline uses a single audio engine: Pocket-TTS, with the
narrator voice. Persona/cloned voices (chatterbox) live on the wizdom branch.

Each voice entry:
- engine: "pocket" (single local TTS engine used by this branch)
- writing_style: id into the voice agent's writing_styles (how the script is written for this voice)
- enabled:      false keeps the voice registered but out of the shuffle
"""

VOICES = {
    # Legacy Pocket-TTS narrator. The one voice this branch uses.
    "narrator": {
        "name": "Narrator (Pocket-TTS)",
        "engine": "pocket",
        "ref_audio": "",
        "params": {},
        "writing_style": "narrator",
        "enabled": True,
    },
}


def get_all_voices() -> dict:
    """Return the full voice registry (including disabled entries)."""
    return VOICES


def get_enabled_voices() -> list:
    """Return (voice_id, voice) tuples for every enabled voice."""
    return [(vid, v) for vid, v in VOICES.items() if v.get("enabled")]


def get_voice(voice_id: str) -> dict:
    """Return a voice entry by id, or None if unknown."""
    return VOICES.get(voice_id)