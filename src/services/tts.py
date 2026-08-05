import os
import re
import numpy as np
import soundfile as sf
from kokoro import KPipeline
from src.config.settings import TEMP_DIR

# Try am_michael if am_adam sounds off on certain lines
KOKORO_VOICE = "am_michael"
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


def generate_audio(lines: list) -> list:
    pipeline = _get_pipeline()
    audio_dir = os.path.join(TEMP_DIR, "audio")

    for line in lines:
        line_id = line["id"]
        text = _clean_text(line["text"])
        path = os.path.join(audio_dir, f"{line_id}.wav")

        print(f"  [tts] Line {line_id}: {text}")

        chunks = []
        for _, _, audio in pipeline(text, voice=KOKORO_VOICE, speed=0.95):
            chunks.append(audio)

        combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sf.write(path, combined, 24000)

        line["audio_path"] = path
        line["actual_duration"] = len(combined) / 24000

    return lines
