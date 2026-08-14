import os
import re

# Set HuggingFace cache dir before importing kokoro so models land in project folder
from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home

import numpy as np
import soundfile as sf
from kokoro import KPipeline
from src.config.settings import TEMP_DIR

# Try am_michael if am_adam sounds off on certain lines
KOKORO_VOICE = "am_adam"
KOKORO_LANG = "a"

_pipeline = None


def _get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code=KOKORO_LANG)
    return _pipeline


def _num_to_words(n: str) -> str:
    """Convert a digit string to spoken words."""
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    try:
        val = int(n)
    except ValueError:
        return n
    if val < 20:
        return ones[val]
    if val < 100:
        return tens[val // 10] + (" " + ones[val % 10] if val % 10 else "")
    if val < 1000:
        rest = val % 100
        return ones[val // 100] + " hundred" + (" " + _num_to_words(str(rest)) if rest else "")
    return n  # fallback for large numbers


def _expand_numbers(text: str) -> str:
    """Expand numbers, symbols and abbreviations to speakable form."""

    # Strip currency symbols before number expansion ($ £ € → spoken by context)
    text = re.sub(r'[$£€]\s*(\d)', r'\1', text)

    # Percentages: 10% → ten percent
    def pct(m):
        return _num_to_words(m.group(1).replace(".", "_")) + " percent" if "." not in m.group(1) else m.group(1) + " percent"
    text = re.sub(r'(\d+\.?\d*)\s*%', pct, text)

    # Decimals: 0.75 → zero point seventy five
    def decimal(m):
        left = _num_to_words(m.group(1)) if m.group(1) != "0" else "zero"
        right = _num_to_words(m.group(2))
        return f"{left} point {right}"
    text = re.sub(r'(\d+)\.(\d+)', decimal, text)

    # Number ranges with hyphen: 3-4 → three to four
    def num_range(m):
        return f"{_num_to_words(m.group(1))} to {_num_to_words(m.group(2))}"
    text = re.sub(r'(\d+)-(\d+)', num_range, text)

    # Thousands with comma: 1,200 → 1200 then let spoken form handle it
    text = re.sub(r'(\d{1,3}),(\d{3})', r'\1\2', text)

    # Remaining plain integers
    text = re.sub(r'\b(\d+)\b', lambda m: _num_to_words(m.group(1)), text)

    # Ordinals
    ordinals = {"1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
                "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
                "9th": "ninth", "10th": "tenth"}
    for k, v in ordinals.items():
        text = re.sub(rf'\b{k}\b', v, text, flags=re.IGNORECASE)

    # Common abbreviations
    abbrevs = {
        r'\bAI\b': 'A I',
        r'\bBMI\b': 'B M I',
        r'\bDNA\b': 'D N A',
        r'\bUV\b': 'U V',
        r'\bHR\b': 'H R',
        r'\bvs\.\b': 'versus',
        r'\be\.g\.\b': 'for example',
        r'\bi\.e\.\b': 'that is',
    }
    for pattern, replacement in abbrevs.items():
        text = re.sub(pattern, replacement, text)

    return text


def _clean_text(text: str) -> str:
    """Normalize text for clean Kokoro TTS output."""
    # Unicode normalization
    text = text.replace("\u2014", ", ").replace("\u2013", ", ")
    text = text.replace("\u2026", "...").replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Expand numbers and symbols before anything else
    text = _expand_numbers(text)

    # Replace connector hyphens with comma
    # Keep: compound words with short glue (waist-to-hip), prefixes (over-, under-, self-)
    SHORT_GLUE = {"to", "a", "an", "of", "the", "in", "on", "at", "by", "up",
                  "do", "go", "be", "is", "or", "as"}
    PREFIXES = {"over", "under", "self", "well", "long", "high", "low", "mid",
                "anti", "pre", "pro", "non", "semi", "ultra", "super", "multi"}

    def replace_connector(m):
        left, right = m.group(1).lower(), m.group(2).lower()
        if left in SHORT_GLUE or right in SHORT_GLUE:
            return m.group(0)
        if left in PREFIXES:  # over-developed, self-aware etc — keep
            return m.group(0)
        if m.group(1)[-1].isdigit() or m.group(2)[0].isdigit():
            return m.group(0)
        if len(left) >= 4 and len(right) >= 4:
            return f"{m.group(1)}, {m.group(2)}"
        return m.group(0)

    text = re.sub(r'(\w+)-(\w+)', replace_connector, text)

    # Ensure ? and ! have space after them
    text = re.sub(r'([?!])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


TARGET_WPM = 150  # natural narration pace — Kokoro at 1.0 ≈ 130 WPM


def _target_speed(text: str) -> float:
    """Calculate speed to hit TARGET_WPM. Clamped to avoid extremes."""
    BASE_WPM = 130
    speed = TARGET_WPM / BASE_WPM
    return max(0.9, min(1.3, speed))


def _noise_gate(audio: np.ndarray, threshold: float = 0.008) -> np.ndarray:
    """Silence samples below threshold to remove background hiss between words."""
    kernel = np.ones(256) / 256
    envelope = np.convolve(np.abs(audio), kernel, mode='same')
    gate = envelope > threshold
    return audio * gate


def generate_audio(lines: list) -> list:
    pipeline = _get_pipeline()
    audio_dir = os.path.join(TEMP_DIR, "audio")

    for line in lines:
        line_id = line["id"]
        text = _clean_text(line["text"])
        path = os.path.join(audio_dir, f"{line_id}.wav")

        print(f"  [tts] Line {line_id}: {text}")

        # Prepend a soft buffer word so Kokoro doesn't clip the first phoneme
        text_with_pause = "Hmm. " + text + "."

        chunks = []
        for _, _, audio in pipeline(text_with_pause, voice=KOKORO_VOICE, speed=_target_speed(text)):
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            chunks.append(audio)

        combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        # Convert to numpy if Kokoro returned a tensor
        if hasattr(combined, 'numpy'):
            combined = combined.numpy()

        # Trim the leading "Hmm" buffer (~0.4s) — keeps the clean start
        trim_samples = int(0.55 * 24000)
        combined = combined[trim_samples:]

        # Append 300ms of silence as buffer
        silence = np.zeros(int(0.3 * 24000), dtype=combined.dtype)
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

        sf.write(path, combined, 24000)

        line["audio_path"] = path
        line["actual_duration"] = len(combined) / 24000

    return lines
