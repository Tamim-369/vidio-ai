import os

# Set HuggingFace cache dir before importing pocket_tts so models land in project folder
from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home

import numpy as np
import soundfile as sf
from src.utils.file_helpers import TEMP_DIR

# Pocket-TTS narrator voice state. If the state file is missing, a reference
# wav can be used to build it (set POCKET_VOICE_REF to the path).
POCKET_VOICE_STATE = "voices/narrator.safetensors"
POCKET_VOICE_REF = os.getenv("POCKET_VOICE_REF", "")

# Normalization + DSP extracted into focused modules; re-exported here so callers
# keep working with the old `from src.services.tts import ...` imports.
from src.services.tts_text import _clean_text
from src.services.tts_dsp import (
    LOUD_GAIN,
    _de_shout,
    postprocess_line,
)

_model = None
_voice_state = None


def _get_pocket_model():
    global _model
    if _model is None:
        from pocket_tts import TTSModel
        _model = TTSModel.load_model()
    return _model


def _get_voice_state():
    global _voice_state
    if _voice_state is None:
        from pocket_tts import export_model_state
        model = _get_pocket_model()
        if os.path.exists(POCKET_VOICE_STATE):
            _voice_state = model.get_state_for_audio_prompt(POCKET_VOICE_STATE)
        elif POCKET_VOICE_REF and os.path.exists(POCKET_VOICE_REF):
            _voice_state = model.get_state_for_audio_prompt(POCKET_VOICE_REF)
            os.makedirs(os.path.dirname(POCKET_VOICE_STATE), exist_ok=True)
            export_model_state(_voice_state, POCKET_VOICE_STATE)
        else:
            raise FileNotFoundError(
                f"No Pocket-TTS narrator voice state at {POCKET_VOICE_STATE} "
                "and no POCKET_VOICE_REF provided."
            )
    return _voice_state


def _generate_pocket(lines: list, audio_dir: str) -> list:
    model = _get_pocket_model()
    voice_state = _get_voice_state()
    sr = model.sample_rate

    for line in lines:
        line_id = line["id"]
        cleaned = _clean_text(line["text"])
        text = _de_shout(cleaned) if line.get("loud") else cleaned
        path = os.path.join(audio_dir, f"{line_id}.wav")

        print(f"  [tts] Line {line_id}: {text}")

        max_tokens = max(200, int(len(text.split()) * 14) + 50)
        audio = model.generate_audio(voice_state, text, max_tokens=max_tokens)

        if hasattr(audio, 'numpy'):
            combined = audio.numpy()
        else:
            combined = np.asarray(audio)

        raw_audio = combined.astype(np.float32)
        gain = LOUD_GAIN if line.get("loud") else 1.0
        final, raw = postprocess_line(
            raw_audio, sr, "pocket", text, {"gain": gain}
        )
        # Preserve pre-finalize "raw" wav beside the final one (pause-only
        # re-bakes rebuild from this without touching the model).
        sf.write(os.path.join(audio_dir, f"{line_id}.raw.wav"), raw, sr)
        sf.write(path, final, sr)
        line["audio_path"] = path
        line["actual_duration"] = len(final) / sr

    return lines


def generate_audio(lines: list, voice: dict = None) -> list:
    """Generate voiceover for each line with Pocket-TTS, written to temp/audio/<id>.wav.

    voice: ignored — the pipeline always uses the Pocket-TTS narrator engine.
    """
    audio_dir = os.path.join(TEMP_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    return _generate_pocket(lines, audio_dir)