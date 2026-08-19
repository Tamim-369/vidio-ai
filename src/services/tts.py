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
    
    if val == 0:
        return "zero"
    if val < 20:
        return ones[val]
    if val < 100:
        return tens[val // 10] + (" " + ones[val % 10] if val % 10 else "")
    if val < 1000:
        rest = val % 100
        result = ones[val // 100] + " hundred"
        if rest:
            # Remove "and" to prevent pauses: "seven hundred thirty one" not "seven hundred and thirty one"
            result += " " + _num_to_words(str(rest))
        return result
    if val < 1000000:
        thousands = val // 1000
        rest = val % 1000
        result = _num_to_words(str(thousands)) + " thousand"
        if rest:
            result += " " + _num_to_words(str(rest))
        return result
    
    return n  # fallback for very large numbers


def _expand_numbers(text: str) -> str:
    """Expand numbers, symbols and abbreviations to speakable form."""

    # CRITICAL: Protect common phrases that should NOT be expanded
    # World War 1/2, F-16, MiG-21, etc.
    protected_patterns = []
    
    # World War I/II/1/2 → Store and restore later
    ww_pattern = r'\b(World War)\s+(I{1,3}|[12])\b'
    def protect_ww(m):
        phrase = m.group(0)
        # Normalize to Roman numerals for speech
        if m.group(2) == '1' or m.group(2) == 'I':
            replacement = "World War One"
        elif m.group(2) == '2' or m.group(2) == 'II':
            replacement = "World War Two"
        else:
            replacement = phrase
        protected_patterns.append((f"___WW{len(protected_patterns)}___", replacement))
        return f"___WW{len(protected_patterns)-1}___"
    
    text = re.sub(ww_pattern, protect_ww, text, flags=re.IGNORECASE)
    
    # Handle standalone WWII, WWI patterns (without "World War" prefix)
    ww_standalone_pattern = r'\b(WW)(I{1,3}|[12])\b'
    def protect_ww_standalone(m):
        roman_or_num = m.group(2)
        if roman_or_num == '1' or roman_or_num == 'I':
            replacement = "World War One"
        elif roman_or_num == '2' or roman_or_num == 'II':
            replacement = "World War Two"
        elif roman_or_num == 'III' or roman_or_num == '3':
            replacement = "World War Three"
        else:
            replacement = f"World War {roman_or_num}"
        protected_patterns.append((f"___WWSA{len(protected_patterns)}___", replacement))
        return f"___WWSA{len(protected_patterns)-1}___"
    
    text = re.sub(ww_standalone_pattern, protect_ww_standalone, text, flags=re.IGNORECASE)
    
    # Handle other common Roman numerals in historical contexts
    roman_pattern = r'\b([A-Z]+)\s+(I{1,4}V?X{0,4}|V|X{1,4})\b'
    def protect_romans(m):
        prefix = m.group(1)
        roman = m.group(2)
        
        # Only process common historical/military contexts
        if prefix.upper() in ['CHAPTER', 'PART', 'SECTION', 'VOLUME', 'BOOK', 'ACT', 'SCENE']:
            # Convert Roman to Arabic number
            roman_to_num = {
                'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
                'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10
            }
            if roman in roman_to_num:
                replacement = f"{prefix} {_num_to_words(str(roman_to_num[roman]))}"
                protected_patterns.append((f"___ROM{len(protected_patterns)}___", replacement))
                return f"___ROM{len(protected_patterns)-1}___"
        
        # Leave other patterns unchanged
        return m.group(0)
    
    text = re.sub(roman_pattern, protect_romans, text)
    
    # Military designations: F-16, MiG-21, B-52, etc.
    # Pattern: Letter(s)-Number(s)
    military_pattern = r'\b([A-Z][a-z]*[A-Z]*)-(\d+[A-Z]?)\b'
    def protect_military(m):
        # F-16 → "F sixteen", MiG-21 → "M i G twenty one"
        letter_part = m.group(1)
        
        # Split into letters with spaces: "MiG" → "M i G", "F" → "F"
        if len(letter_part) == 1:
            letter = letter_part
        elif letter_part.isupper():
            # All caps like "USA" → "U S A"
            letter = ' '.join(letter_part)
        else:
            # Mixed case like "MiG" → split all letters
            letter = ' '.join(c for c in letter_part)
        
        num = m.group(2)
        # Extract number and possible letter suffix
        num_match = re.match(r'(\d+)([A-Z]?)', num)
        if num_match:
            number = int(num_match.group(1))
            suffix = num_match.group(2)
            number_words = _num_to_words(str(number))
            replacement = f"{letter} {number_words}"
            if suffix:
                replacement += f" {suffix}"
        else:
            replacement = f"{letter} {num}"
        protected_patterns.append((f"___MIL{len(protected_patterns)}___", replacement))
        return f"___MIL{len(protected_patterns)-1}___"
    
    text = re.sub(military_pattern, protect_military, text)
    
    # Year ranges: "1939 to 1945" or "1939-1945" (MUST come before single years)
    year_range_pattern = r'\b(\d{4})\s+(to|-)\s+(\d{4})\b'
    def protect_year_range(m):
        year1 = m.group(1)
        connector = " to " if m.group(2) == "to" else " to "
        year2 = m.group(3)
        
        def format_year(year_str):
            year_int = int(year_str)
            if 1000 <= year_int <= 1999:
                century = year_int // 100
                remainder = year_int % 100
                if remainder == 0:
                    return _num_to_words(str(century)) + " hundred"
                else:
                    return _num_to_words(str(century)) + " " + _num_to_words(str(remainder))
            elif 2000 <= year_int <= 2009:
                return "two thousand" + (" " + _num_to_words(str(year_int % 2000)) if year_int % 2000 else "")
            elif year_int >= 2010:
                century = year_int // 100
                remainder = year_int % 100
                return _num_to_words(str(century)) + " " + _num_to_words(str(remainder))
            else:
                return _num_to_words(year_str)
        
        replacement = format_year(year1) + connector + format_year(year2)
        protected_patterns.append((f"___YRNG{len(protected_patterns)}___", replacement))
        return f"___YRNG{len(protected_patterns)-1}___"
    
    text = re.sub(year_range_pattern, protect_year_range, text)
    
    # Years in context: "in 1945", "since 1960", "the 1980s"
    # Keep years as numbers for natural speech
    year_pattern = r'\b(in|since|during|from|until|circa|around|about)\s+(\d{4}s?)\b'
    def protect_years(m):
        prefix = m.group(1)
        year = m.group(2)
        # 1945 → "nineteen forty five", 2001 → "two thousand one"
        if year.endswith('s'):
            year_num = year[:-1]
            is_plural = True
        else:
            year_num = year
            is_plural = False
        
        year_int = int(year_num)
        if 1000 <= year_int <= 1999:
            century = year_int // 100
            remainder = year_int % 100
            if remainder == 0:
                year_words = _num_to_words(str(century)) + " hundred"
            else:
                year_words = _num_to_words(str(century)) + " " + _num_to_words(str(remainder))
        elif 2000 <= year_int <= 2009:
            year_words = "two thousand " + (_num_to_words(str(year_int % 2000)) if year_int % 2000 else "")
        elif year_int >= 2010:
            century = year_int // 100
            remainder = year_int % 100
            year_words = _num_to_words(str(century)) + " " + _num_to_words(str(remainder))
        else:
            year_words = _num_to_words(year_num)
        
        if is_plural:
            year_words += "s"
        
        replacement = f"{prefix} {year_words}"
        protected_patterns.append((f"___YEAR{len(protected_patterns)}___", replacement))
        return f"___YEAR{len(protected_patterns)-1}___"
    
    text = re.sub(year_pattern, protect_years, text, flags=re.IGNORECASE)
    
    # Decades: "the 1980s", "early 2000s"
    decade_pattern = r'\b(the|early|late|mid)\s+(\d{4})s\b'
    def protect_decades(m):
        prefix = m.group(1)
        decade = m.group(2)
        decade_int = int(decade)
        
        if 1000 <= decade_int <= 1999:
            century = decade_int // 100
            remainder = decade_int % 100
            if remainder == 0:
                decade_words = _num_to_words(str(century)) + " hundreds"
            else:
                # Handle "eighties" not "eightys"
                num_word = _num_to_words(str(remainder))
                if num_word.endswith('y'):
                    decade_words = _num_to_words(str(century)) + " " + num_word[:-1] + "ies"
                else:
                    decade_words = _num_to_words(str(century)) + " " + num_word + "s"
        elif decade_int >= 2000:
            if decade_int == 2000:
                decade_words = "two thousands"
            else:
                century = decade_int // 100
                remainder = decade_int % 100
                num_word = _num_to_words(str(remainder))
                if num_word.endswith('y'):
                    decade_words = _num_to_words(str(century)) + " " + num_word[:-1] + "ies"
                else:
                    decade_words = _num_to_words(str(century)) + " " + num_word + "s"
        else:
            decade_words = _num_to_words(decade) + "s"
        
        replacement = f"{prefix} {decade_words}"
        protected_patterns.append((f"___DEC{len(protected_patterns)}___", replacement))
        return f"___DEC{len(protected_patterns)-1}___"
    
    text = re.sub(decade_pattern, protect_decades, text, flags=re.IGNORECASE)

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

    # Number ranges with hyphen: 3-4 → three to four (but not military designations)
    def num_range(m):
        return f"{_num_to_words(m.group(1))} to {_num_to_words(m.group(2))}"
    text = re.sub(r'(\d+)-(\d+)', num_range, text)

    # Thousands with comma: 1,200 → 1200 then let spoken form handle it
    text = re.sub(r'(\d{1,3}),(\d{3})', r'\1\2', text)

    # Remaining plain integers
    text = re.sub(r'\b(\d+)\b', lambda m: _num_to_words(m.group(1)), text)

    # Ordinals - expand these more thoroughly
    ordinals = {
        "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
        "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
        "9th": "ninth", "10th": "tenth", "11th": "eleventh", "12th": "twelfth",
        "13th": "thirteenth", "14th": "fourteenth", "15th": "fifteenth",
        "16th": "sixteenth", "17th": "seventeenth", "18th": "eighteenth",
        "19th": "nineteenth", "20th": "twentieth", "21st": "twenty first",
        "22nd": "twenty second", "23rd": "twenty third", "30th": "thirtieth",
        "40th": "fortieth", "50th": "fiftieth", "100th": "hundredth"
    }
    for k, v in ordinals.items():
        text = re.sub(rf'\b{k}\b', v, text, flags=re.IGNORECASE)

    # Common abbreviations
    abbrevs = {
        r'\bAI\b': 'A I',
        r'\bBMI\b': 'B M I',
        r'\bDNA\b': 'D N A',
        r'\bUV\b': 'U V',
        r'\bHR\b': 'H R',
        r'\bUSA\b': 'U S A',
        r'\bUK\b': 'U K',
        r'\bUSSR\b': 'U S S R',
        r'\bNATO\b': 'N A T O',
        r'\bRAF\b': 'R A F',
        r'\bUSAF\b': 'U S A F',
        r'\bvs\.\b': 'versus',
        r'\bvs\b': 'versus',
        r'\be\.g\.\b': 'for example',
        r'\bi\.e\.\b': 'that is',
        r'\bMPH\b': 'miles per hour',
        r'\bmph\b': 'miles per hour',
        r'\bKPH\b': 'kilometers per hour',
        r'\bkph\b': 'kilometers per hour',
    }
    for pattern, replacement in abbrevs.items():
        text = re.sub(pattern, replacement, text)
    
    # Restore protected patterns
    for placeholder, replacement in protected_patterns:
        text = text.replace(placeholder, replacement)

    return text


def _clean_text(text: str) -> str:
    """Normalize text for clean Kokoro TTS output with natural pacing."""
    # Unicode normalization
    text = text.replace("\u2014", ", ").replace("\u2013", ", ")
    text = text.replace("\u2026", "...").replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Expand numbers and symbols before anything else
    text = _expand_numbers(text)
    
    # Add strategic pauses for natural speech flow
    
    # 1. Add pauses after list numbers
    text = re.sub(r'^(number\s+\w+)([\.\:\,\!])', r'\1,,', text, flags=re.IGNORECASE)
    text = re.sub(r'^(\w+)(\.)\s+', r'\1,, ', text)  # "Ten. Something" → "Ten,, Something"
    text = re.sub(r'\b(number\s+\w+)([\.\:\!])', r'\1,,', text, flags=re.IGNORECASE)
    text = re.sub(r'^(number\s+\w+)(\s+)', r'\1,,\2', text, flags=re.IGNORECASE)
    
    # 2. Add pauses after introductory phrases (but NOT mid-sentence conjunctions)
    intro_phrases = [
        r'^(however|meanwhile|furthermore|moreover|additionally|consequently)\b',
        r'^(in fact|for example|for instance|in particular|notably)\b',
        r'^(first|second|third|finally|lastly|initially)\b',
        r'^(during|after|before|since|until)\b',  # Only at sentence start
        r'^(despite|although|even though|whereas)\b'
    ]
    for pattern in intro_phrases:
        text = re.sub(pattern, r'\1,', text, flags=re.IGNORECASE)
    
    # 3. Add pauses after prepositional phrases (when they start sentences)
    text = re.sub(r'^(in \w+|during \w+|after \w+|before \w+|with \w+)', r'\1,', text, flags=re.IGNORECASE)
    
    # 4. Add pauses around parenthetical information
    text = re.sub(r'\(([^)]+)\)', r', \1,', text)  # (info) → , info,
    
    # 5. Ensure proper pauses around commas and semicolons
    text = re.sub(r'([,;])\s*', r'\1 ', text)  # Ensure space after commas/semicolons
    
    # 6. Add pauses before "and", "but", "or" when they connect clauses (but NOT mid-sentence conjunctions)
    # Only add pauses for clause-connecting conjunctions at natural break points
    text = re.sub(r'([,;])\s+(and|but|or)\s+', r'\1 \2 ', text)  # After existing punctuation only
    
    # 7. Add breathing pauses in long sentences (every 15-20 words)
    words = text.split()
    if len(words) > 18:  # Increased threshold to reduce over-pausing
        # Insert a pause roughly in the middle of long sentences
        mid_point = len(words) // 2
        # Find a natural break point (after comma, semicolon, etc.) - NOT after conjunctions
        for i in range(mid_point - 3, min(mid_point + 4, len(words))):
            if i < len(words) and words[i].endswith((',', ';')):
                words[i] = words[i] + ','  # Double the pause
                break
        text = ' '.join(words)

    # Replace connector hyphens with comma (existing logic)
    SHORT_GLUE = {"to", "a", "an", "of", "the", "in", "on", "at", "by", "up",
                  "do", "go", "be", "is", "or", "as"}
    PREFIXES = {"over", "under", "self", "well", "long", "high", "low", "mid",
                "anti", "pre", "pro", "non", "semi", "ultra", "super", "multi"}

    def replace_connector(m):
        left, right = m.group(1).lower(), m.group(2).lower()
        if left in SHORT_GLUE or right in SHORT_GLUE:
            return m.group(0)
        if left in PREFIXES:
            return m.group(0)
        if m.group(1)[-1].isdigit() or m.group(2)[0].isdigit():
            return m.group(0)
        if len(left) >= 4 and len(right) >= 4:
            return f"{m.group(1)}, {m.group(2)}"
        return m.group(0)

    text = re.sub(r'(\w+)-(\w+)', replace_connector, text)

    # Ensure proper spacing after punctuation
    text = re.sub(r'([?!])([^\s])', r'\1 \2', text)
    text = re.sub(r'([.])([A-Z])', r'\1 \2', text)  # Space after periods before capitals
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


TARGET_WPM = 130  # Consistent natural pace - matches Kokoro base for stability


def _target_speed(text: str) -> float:
    """Calculate speed to hit TARGET_WPM with consistent pacing."""
    BASE_WPM = 130  # Kokoro's base speed
    target_speed = TARGET_WPM / BASE_WPM
    
    # Less aggressive speed adjustment - keep more consistent
    word_count = len(text.split())
    if word_count > 25:
        # Only slow down very long sentences slightly
        target_speed *= 0.92  # 8% slower (was 15% slower)
    elif word_count > 20:
        target_speed *= 0.95  # 5% slower (was 10% slower)
    # Remove the 15-word threshold that was causing inconsistency
    
    # Tighter range to prevent extreme speeds
    return max(0.85, min(1.05, target_speed))  # 0.85-1.05 instead of 0.7-1.1


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

        # Use a simpler buffer that's easier to trim cleanly
        text_with_pause = "Hello. " + text

        chunks = []
        for _, _, audio in pipeline(text_with_pause, voice=KOKORO_VOICE, speed=_target_speed(text)):
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            chunks.append(audio)

        combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        # Convert to numpy if Kokoro returned a tensor
        if hasattr(combined, 'numpy'):
            combined = combined.numpy()

        # Trim the leading "Hello." buffer more precisely (~0.8s)
        trim_samples = int(0.8 * 24000)
        combined = combined[trim_samples:]

        # Don't add any trailing buffer - let the sentence end naturally
        # Add very minimal silence between sentences - just enough for natural breathing
        silence = np.zeros(int(0.05 * 24000), dtype=combined.dtype)  # 50ms gap (barely noticeable)
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
