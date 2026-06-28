import os
from supertonic import TTS
from src.config.settings import TEMP_DIR

VOICE_NAME = "M3"
VOICE_LANG = "en"
VOICE_STEPS = 12      # Quality: 5 (low) to 12 (high)
VOICE_SPEED = 0.8   # Speed: 0.7 (slow) to 2.0 (fast)

_tts_instance = None
_voice_style = None


def _get_tts() -> TTS:
    """Lazy-load the TTS model and voice style (downloaded once from HuggingFace, reused across calls)."""
    global _tts_instance, _voice_style
    if _tts_instance is None:
        print("  [tts] Loading Supertonic model (first run may download from HuggingFace)...")
        _tts_instance = TTS(auto_download=True)
        _voice_style = _tts_instance.get_voice_style(voice_name=VOICE_NAME)
    return _tts_instance


def generate_audio(lines: list) -> list:
    """
    Generate one WAV file per line using Supertonic TTS.
    Returns lines with audio_path + actual_duration populated.
    """
    audio_dir = os.path.join(TEMP_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    tts = _get_tts()

    for line in lines:
        line_id = line["id"]
        text = line["text"]
        path = os.path.join(audio_dir, f"{line_id}.wav")

        print(f"  [tts] Generating audio for line {line_id}: '{text[:60]}...'")

        wav, duration = tts.synthesize(
            text=text,
            lang=VOICE_LANG,
            voice_style=_voice_style,
            total_steps=VOICE_STEPS,
            speed=VOICE_SPEED,
        )

        tts.save_audio(wav, path)

        line["audio_path"] = path
        line["actual_duration"] = float(duration[0])

        print(f"  [tts] Done — {line['actual_duration']:.2f}s saved to {path}")

    return lines