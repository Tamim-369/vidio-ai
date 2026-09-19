"""Voice registry — data-driven list of voices available to the video generator.

Each voice entry:
- engine: "chatterbox" (real-voice cloning, CPU) | "pocket" (legacy narrator)
- ref_audio: path to the reference clip used to clone the voice (chatterbox only)
- params:       engine-specific TTS knobs (exaggeration/cfg_weight/temperature for chatterbox)
- writing_style: id into src/config/writing_styles.py (how the script is written for this voice)
- enabled:      false keeps the voice registered but out of the shuffle

To add a new voice: drop a clean reference WAV somewhere under src/voices_to_clone/
and add one entry here (copy a commented template below).
"""

VOICES = {
    # --- Template (uncomment to add a voice) ---
    # "new_voice": {
    #     "name": "New Voice",
    #     "engine": "chatterbox",
    #     "ref_audio": "src/voices_to_clone/voice_candidates/myvoice_ref.wav",
    #     "params": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
    #     "writing_style": "narrator",
    #     "enabled": False,
    # },

    "donald-trump": {
        "name": "Donald Trump",
        "engine": "chatterbox",
        "ref_audio": "src/voices_to_clone/candidates/donald-trump/donald-trump_ref.wav",
        # Lower temp = stable speaker identity/emotion across lines; the script's
        # own "loud" lines still get a deterministic volume emphasis in TTS.
        "params": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.75, "gain": 1.15,
                   "eq": ["highpass 100", "equalizer 3000 1 2.5", "equalizer 6500 1 1.5"],
                   "speed": 1.0},
        "writing_style": "trump",
        "enabled": True,
    },

    "arnold-schwarzenegger": {
        "name": "Arnold Schwarzenegger",
        "engine": "chatterbox",
        "ref_audio": "src/voices_to_clone/candidates/arnold-schwarzenegger/arnold-schwarzenegger_ref.wav",
        # Lower exaggeration + temperature → emotional but consistent, no random
        # angry swings between lines; loud peaks come from the TTS loud flag.
        "params": {"exaggeration": 0.65, "cfg_weight": 0.7, "temperature": 0.72, "gain": 1.02,
                   "speed": 1.0},
        "writing_style": "arnold",
        "enabled": True,
    },

    # Legacy Pocket-TTS narrator. Out of the round-robin by default — re-enable
    # (enabled: True) to give it a turn in the voice cycle alongside the clones.
    "narrator": {
        "name": "Narrator (Pocket-TTS)",
        "engine": "pocket",
        "ref_audio": "",
        "params": {},
        "writing_style": "narrator",
        "enabled": False,
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