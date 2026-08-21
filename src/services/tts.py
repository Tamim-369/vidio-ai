import os
import re

# Set HuggingFace cache dir before importing pocket_tts so models land in project folder
from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home

import numpy as np
import soundfile as sf
from pocket_tts import TTSModel, export_model_state
from src.config.settings import TEMP_DIR, POCKET_VOICE_STATE, POCKET_VOICE_REF

_model = None
_voice_state = None


def _get_model() -> TTSModel:
    global _model
    if _model is None:
        _model = TTSModel.load_model()
    return _model


def _get_voice_state() -> dict:
    global _voice_state
    if _voice_state is None:
        model = _get_model()
        if os.path.exists(POCKET_VOICE_STATE):
            _voice_state = model.get_state_for_audio_prompt(POCKET_VOICE_STATE)
        else:
            _voice_state = model.get_state_for_audio_prompt(POCKET_VOICE_REF)
            os.makedirs(os.path.dirname(POCKET_VOICE_STATE), exist_ok=True)
            export_model_state(_voice_state, POCKET_VOICE_STATE)
    return _voice_state


def _clean_text(text: str) -> str:
    """Normalize text for clean Pocket-TTS output."""
    # Unicode normalization
    text = text.replace("\u2014", ", ").replace("\u2013", ", ")
    text = text.replace("\u2026", "...").replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Common abbreviations Pocket-TTS may mangle
    abbrevs = {
        r'\bAI\b': 'A I',
        r'\bDNA\b': 'D N A',
        r'\bUSA\b': 'U S A',
        r'\bUK\b': 'U K',
        r'\bUSSR\b': 'U S S R',
        r'\bNATO\b': 'N A T O',
        r'\bRAF\b': 'R A F',
        r'\bUSAF\b': 'U S A F',
        r'\be\.g\.\b': 'for example',
        r'\bi\.e\.\b': 'that is',
        r'\bvs\.\b': 'versus',
        r'\bvs\b': 'versus',
        r'\bMPH\b': 'miles per hour',
        r'\bmph\b': 'miles per hour',
        r'\bKPH\b': 'kilometers per hour',
        r'\bkph\b': 'kilometers per hour',
    }
    for pattern, replacement in abbrevs.items():
        text = re.sub(pattern, replacement, text)

    # Ensure proper spacing after punctuation
    text = re.sub(r'([?!])([^\s])', r'\1 \2', text)
    text = re.sub(r'([.])([A-Z])', r'\1 \2', text)  # Space after periods before capitals

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def _noise_gate(audio: np.ndarray, threshold: float = 0.008) -> np.ndarray:
    """Silence samples below threshold to remove background hiss between words."""
    kernel = np.ones(256) / 256
    envelope = np.convolve(np.abs(audio), kernel, mode='same')
    gate = envelope > threshold
    return audio * gate


def generate_audio(lines: list) -> list:
    model = _get_model()
    voice_state = _get_voice_state()
    audio_dir = os.path.join(TEMP_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    sr = model.sample_rate

    for line in lines:
        line_id = line["id"]
        text = _clean_text(line["text"])
        path = os.path.join(audio_dir, f"{line_id}.wav")

        print(f"  [tts] Line {line_id}: {text}")

        max_tokens = max(200, int(len(text.split()) * 14) + 50)
        audio = model.generate_audio(voice_state, text, max_tokens=max_tokens)

        if hasattr(audio, 'numpy'):
            combined = audio.numpy()
        else:
            combined = np.asarray(audio)

        # Add very minimal silence between sentences - just enough for natural breathing
        silence = np.zeros(int(0.05 * sr), dtype=combined.dtype)  # 50ms gap (barely noticeable)
        combined = np.concatenate([combined, silence])

        # Apply noise gate to remove hiss between words
        combined = _noise_gate(combined)

        # RMS normalization — consistent loudness across all lines
        # Target RMS level (0.0 to 1.0) — 0.15 is a good natural speaking level
        TARGET_RMS = 0.15
        rms = np.sqrt(np.mean(combined ** 2))
        if rms > 0:
            combined = combined * (TARGET_RMS / rms)
        # Hard clip to prevent any peaks from distorting
        combined = np.clip(combined, -0.95, 0.95)

        sf.write(path, combined, sr)

        line["audio_path"] = path
        line["actual_duration"] = len(combined) / sr

    return lines
