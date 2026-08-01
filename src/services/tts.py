import os
import re
import numpy as np
import soundfile as sf
from kokoro import KPipeline
from src.config.settings import TEMP_DIR

# American English, deep male voice
# Other good male voices: am_adam, am_michael
KOKORO_VOICE = "am_adam"
KOKORO_LANG = "a"  # 'a' = American English

_pipeline = None


def _get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code=KOKORO_LANG)
    return _pipeline


def _clean_text(text: str) -> str:
    """Normalize text for clean TTS output."""
    # Unicode normalization
    text = text.replace("\u2014", ", ").replace("\u2013", ", ")
    text = text.replace("\u2026", "...").replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Replace hyphens used as sentence connectors with a comma.
    # Keeps legitimate compounds (waist-to-hip, over-developed, 3-4)
    # by only replacing when both sides are long standalone words.
    SHORT_GLUE = {"to", "a", "an", "of", "the", "in", "on", "at", "by", "up",
                  "do", "go", "be", "is", "or", "as"}

    def replace_connector(m):
        left, right = m.group(1).lower(), m.group(2).lower()
        # Keep if either side is a short glue word or a digit
        if left in SHORT_GLUE or right in SHORT_GLUE:
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


def generate_audio(lines: list) -> list:
    pipeline = _get_pipeline()
    audio_dir = os.path.join(TEMP_DIR, "audio")

    for line in lines:
        line_id = line["id"]
        text = _clean_text(line["text"])
        path = os.path.join(audio_dir, f"{line_id}.wav")

        print(f"  [tts] Generating audio for line {line_id}")

        chunks = []
        for _, _, audio in pipeline(text, voice=KOKORO_VOICE, speed=1.0):
            chunks.append(audio)

        combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sf.write(path, combined, 24000)

        line["audio_path"] = path
        line["actual_duration"] = len(combined) / 24000

    return lines
