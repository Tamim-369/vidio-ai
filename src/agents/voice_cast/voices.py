"""Voice registry: the voices the generator can narrate with.

Per entry:
- engine        "chatterbox" (voice cloning) | "pocket" (legacy narrator)
- ref_audio     reference clip to clone from (chatterbox only)
- params        engine knobs (exaggeration/cfg_weight/temperature)
- writing_style id into writing_styles.py
- face          full-bleed card background. The face already sits bottom-right
                in these photos, which is why the quote is typeset top-left and
                the byline bottom-left.
- quote_authors fake byline names, e.g. "Don Tzu". Seeded per video so the
                byline is not always the same. "|" splits quoted form from
                book/work form: '"Don Tzu" | "Fart of War"'.
- enabled       false keeps the voice registered but out of the shuffle

To add one: drop a clean reference WAV under
src/experiments/voices_to_clone/ and copy the template below.
"""

VOICES = {
    # --- Template (uncomment to add a voice) ---
    # "new_voice": {
    #     "name": "New Voice",
    #     "engine": "chatterbox",
    #     "ref_audio": "src/experiments/voices_to_clone/voice_candidates/myvoice_ref.wav",
    #     "params": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
    #     "writing_style": "narrator",
    #     "face": "src/faces/NewVoice.png",
    #     "quote_authors": ["Anon | Sayings"],
    #     "enabled": False,
    # },

    "donald-trump": {
        "name": "Donald Trump",
        "engine": "chatterbox",
        "ref_audio": "src/experiments/voices_to_clone/candidates/donald-trump/donald-trump_ref.wav",
        # Lower temp = stable speaker identity/emotion across lines; the script's
        # own "loud" lines still get a deterministic volume emphasis in TTS.
        # Reduced EQ boosts to prevent hiss fog; removed 6.5k boost entirely.
        "params": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.75, "gain": 1.1,
                   "eq": ["highpass 100", "equalizer 3000 1 1.5"],
                   "speed": 1.0},
        "writing_style": "trump",
        # The joke prompt is written about this subject, so the joke and the
        # narrator always agree. Rotating the voice rotates the subject.
        "subject": "War and military strategy",
        "face": "src/faces/Trump.png",
        "quote_authors": [
            "Don Tzu | The Fart of War",
        ],
        "enabled": True,
    },

    "arnold-schwarzenegger": {
        "name": "Arnold Schwarzenegger",
        "engine": "chatterbox",
        "ref_audio": "src/experiments/voices_to_clone/candidates/arnold-schwarzenegger/arnold-schwarzenegger_ref.wav",
        # Measured sweeps (word-end pitch droop + identity drift on 24k ref):
        # final pick = exag 0.65 / cfg 0.9 / temp 0.55 with a 72s ref of 37 clean
        # vocals-stem segments. cfg 0.85–0.9 anchors identity through line ends;
        # temp 0.5–0.55 keeps the delivery steady without robotic monotone.
        "params": {"exaggeration": 0.65, "cfg_weight": 0.9, "temperature": 0.55, "gain": 1.02,
                   "repetition_penalty": 1.2, "min_p": 0.05, "top_p": 1.0,
                   "speed": 1.0},
        "writing_style": "arnold",
        "subject": "Weight lifting and bodybuilding",
        "face": "src/faces/Arnold.png",
        "quote_authors": [
            "Brolexander | The Book of Gainz",
        ],
        # Back in rotation: his prompt gained the same funny-gate and
        # concrete-detail rules that fixed Andru Tatte, so the quotes are no
        # longer gym metaphors with sincere explanations.
        "enabled": True,
    },

    "andrew-tate": {
        "name": "Andrew Tate",
        "engine": "chatterbox",
        "ref_audio": "src/experiments/voices_to_clone/candidates/andrew-tate/andrew-tate_ref.wav",
        # A/B selected: prime 103.5–114.0s (119Hz conversational register).
        # Mod-low temp keeps the aggro-but-composed take.
        "params": {"exaggeration": 0.65, "cfg_weight": 0.85, "temperature": 0.55, "gain": 1.02,
                   "repetition_penalty": 1.2, "min_p": 0.05, "top_p": 1.0,
                   "speed": 1.0},
        "writing_style": "andrew_tate",
        # Not money: he is about life, self-improvement and hard work. This only
        # labels the video, the prompt itself takes no subject.
        "subject": "Life and self-improvement",
        "face": "src/faces/Tate.png",
        "quote_authors": [
            "Andru Tatte | The Way of Whatever",
        ],
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


def pick_quote_author(voice: dict, seed: str = "") -> tuple:
    """Return (attribution, source) for a fake quote credit on the card.

One voice speaks under many fake authors, so the byline is derived from the
quote text (stable per quote) rather than fixed per voice. An entry's "|" splits
the spoken name from the book/work name; without it both halves are identical.
Falls back to ("", "") for voices with no author list."""
    authors = voice.get("quote_authors") or []
    if not authors:
        return "", ""
    key = seed or voice.get("name", "")
    pick = authors[sum(ord(c) for c in key) % len(authors)]
    if "|" in pick:
        name, source = (part.strip() for part in pick.split("|", 1))
    else:
        name = source = pick.strip()
    return name, source