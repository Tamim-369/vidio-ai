import json
import os
import re
from src.config.prompt import get_raw_script_prompt
from src.config.settings import TEMP_DIR

# LLM plumbing (client lifecycle, retries, Cloudflare/Groq fallback) lives in llm.py.
from src.services.llm import call_text

# Call seam: Cloudflare primary, Groq secondary.
_call_text = call_text

# Prompt for converting raw script to structured JSON
JSON_STRUCTURE_PROMPT = """You are a JSON converter. Your ONLY job is to convert a raw video script into structured JSON.

IMPORTANT: You MUST preserve EVERY line from the raw script. Do not skip any lines.

The raw script will have numbered lines like:
1. First sentence here.
2. Second sentence here.

You must convert EACH numbered line into a JSON object.

Return ONLY valid JSON. No explanation, no markdown, no code blocks. Just raw JSON.

Format:
{
  "topic": "<topic>",
  "lines": [
    {
      "id": 1,
      "text": "<original text from line 1, without the number prefix>",
      "search_term": "<3-8 word search query>",
      "image_expectation": "<15-20 word visual description>",
      "image_type": "<stock|search>",
      "duration": <3-7 seconds as integer>,
      "loud": false
    },
    {
      "id": 2,
      "text": "<original text from line 2, without the number prefix>",
      "search_term": "<3-8 word search query>",
      "image_expectation": "<15-20 word visual description>",
      "image_type": "<stock|search>",
      "duration": <3-7 seconds as integer>,
      "loud": false
    }
  ]
}

Rules:
- Create one object per numbered line from the raw script
- text: The original sentence WITHOUT the number prefix (e.g., "1. " or "2. ")
- search_term: 2-5 words MAX of a REAL, PHOTOGRAPHABLE subject that an image search would actually return. Use real places, monuments, memorials, battle sites, named historical events, period photos, real equipment, museums, reenactments, maps. NEVER turn a story into keywords ("1,200 km concrete beast" is unusable), NEVER use metaphors/abstract concepts, NEVER lead with numbers instead of the subject. If the moment has no specific real subject, pick a real adjacent generic scene that stock sites have (e.g. "WW2 Russian front winter" not "3.3 million men on the front"). Example: 'Maginot Line' not '1200 km concrete fortifications'; 'Gallipoli 1915 landing' not '400000 troops marched into peninsula'.
- image_expectation: 12-20 words describing the concrete PHOTOGRAPHIC SUBJECT the camera should see — foreground, setting, mood. It MUST be a real, photographable scene, not a fantasy: no ghosts, holo-overlays, weight bars, or impossible compositions. Describe what an actual photo of this thing looks like.
- image_type: "search" when a specific named real thing or period photo exists (a monument, battle site, artifact, historical photo). "stock" only for generic atmospheric scenes (snow, fog, empty landscape, flags, crowds) that clearly exist as stock photos
- duration: How long it takes to speak (3-7 seconds)
- loud: true ONLY for explosive, dramatic, anger or exclamatory lines that should be SHOUTED (e.g. "they burned them alive!!"). false for normal narration. Default false, and only raise the volume if the sentence genuinely calls for it.
- Do NOT skip any lines - convert ALL of them"""


def _normalize_raw(raw: str) -> str:
    """Strip markdown blocks and normalize Unicode punctuation."""
    if not raw:
        return raw

    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

# Remove thinking tags
    raw = re.sub(r' thinking.*?response', '', raw, flags=re.DOTALL)
    raw = re.sub(r'<thinking>.*?</thinking>', '', raw, flags=re.DOTALL)
    raw = raw.strip()

    # Normalize Unicode punctuation
    replacements = {
        "\u2011": "-", "\u2013": "-", "\u2014": "-",
        "\u2015": "-", "\u2212": "-",
        "\u2018": "'", "\u2019": "'", "\u201A": "'",
        "\u201C": '"', "\u201D": '"', "\u201E": '"',
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        raw = raw.replace(src, dst)

    return raw


# Catchphrases often tacked on to the END of a spoken line by the LLM when
# channeling a persona (Trump especially) — never necessary, always filler.
_FILLER_TAILS = (
    "believe me",
    "listen to me",
    "let me tell you",
    "folks",
    "i mean it",
    "trust me",
    "you know what",
    "you know",
    "let me be clear",
    "plain and simple",
    "no question about it",
)


def _strip_ending_filler(text: str) -> str:
    """Remove a filler catchphrase glued to the END of a sentence/line.

    Targets "Believe me." / "… believe me" as an unneeded trailing tag, not the
    same words used mid-sentence where they might be story-level. Applied per
    line so blank lines / number prefixes are preserved.
    """
    out_lines = []
    for ln in (text or "").splitlines():
        if not ln.strip():
            out_lines.append(ln)
            continue
        stripped = ln.strip()
        while True:
            lowered = stripped.lower()
            matched = False
            for phrase in _FILLER_TAILS:
                m = re.search(
                    r"(?:[,.;!?:…-]|\s|^)\s*" + re.escape(phrase) + r"\s*([.!?…]*)\s*$",
                    lowered,
                )
                if m:
                    head = stripped[: m.start()].rstrip(" ,.;!?:…-")
                    trailing = m.group(1)
                    stripped = (head.rstrip(" \t-") + trailing).strip()
                    matched = True
                    break
            if not matched:
                break
        out_lines.append(stripped)
    return "\n".join(out_lines)


def _missing_closers(text: str) -> str:
    """If `text` was truncated by the LLM, return the braces/brackets needed to
    close the still-open scopes, in the right nesting order. Returns '' when the
    scopes are balanced (or text is broken, not just truncated)."""
    stack = []
    in_str, esc = False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            pair = {"}": "{", "]": "["}.get(ch)
            if stack and stack[-1] == pair:
                stack.pop()
    closing = {"{": "}", "[": "]"}
    return "".join(closing[c] for c in reversed(stack))


def _safe_json_loads(text: str) -> dict:
    """Parse JSON with common LLM-output repairs (bad inner quotes, trailing
    commas, missing commas, and mid-string truncation)."""
    def attempts(candidate):
        for transform in (
            lambda s: s,
            lambda s: re.sub(r",\s*([}\]])", r"\1", s),
            lambda s: re.sub(r"\}\s*\{", "},{", s),
            lambda s: re.sub(r"\]\s*\[", "],[", s),
            lambda s: s + _missing_closers(s),
            lambda s: re.sub(r",\s*([}\]])", r"\1", s) + _missing_closers(re.sub(r",\s*([}\]])", r"\1", s)),
            _recover_truncated,
        ):
            try:
                return json.loads(transform(candidate))
            except (json.JSONDecodeError, ValueError):
                continue
        raise json.JSONDecodeError("unrepairable JSON", candidate, 0)

    try:
        return attempts(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the doc's outermost {...} block (helps when the LLM wrapped
    # the JSON in prose/markdown).
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return attempts(candidate)
        except json.JSONDecodeError:
            text = candidate

    def fix_inner_quotes(m):
        key, value = m.group(1), m.group(2)
        value = re.sub(r'(?<!\\)"', "'", value)
        return f'"{key}": "{value}"'

    repaired = re.sub(r'"(\w+)":\s*"(.*?)"(?=\s*[,}\]])', fix_inner_quotes, text, flags=re.DOTALL)

    try:
        return attempts(repaired)
    except json.JSONDecodeError as e:
        ln = getattr(e, "lineno", 0)
        lines = repaired.splitlines()
        snippet = lines[ln - 1] if 0 < ln <= len(lines) else "<eof>"
        raise json.JSONDecodeError(f"{e.msg}: {snippet!r}", repaired, e.pos) from e


def _recover_truncated(text: str) -> str:
    """Try salvaging a truncated response: chop a small unterminated tail
    (e.g. a string cut mid-word) and close the remaining scopes. Returns the
    repaired JSON text, or raises if no chop repairs it."""
    for k in range(64):
        if k == 0:
            continue
        c = text[:-k] if k < len(text) else ""
        if not c or c[-1] not in '"}]0123456789truefals':
            continue
        repaired = c + _missing_closers(c)
        try:
            json.loads(repaired)
            return repaired
        except (json.JSONDecodeError, ValueError):
            continue
    raise json.JSONDecodeError("unrepairable truncated JSON", text, 0)


# NOTE: LLM call helpers are defined in src/services/llm.py.
# `_call_text` above is the stat-kept seam used by script generation.


def _extract_numbered_script(text: str) -> str:
    """Keep only the numbered script lines, dropping any reasoning/preamble the
    model emitted before line 1 or commentary tacked on after the last line."""
    lines = (text or "").splitlines()
    start = end = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*\d{1,3}\s*[.)]", ln):
            if start is None:
                start = i
            end = i + 1
    if start is None:
        return text
    return "\n".join(lines[start:end])


def _generate_raw_script(topic: str, raw_data: str, style: dict = None) -> str:
    """Generate raw spoken script using the viral script prompt."""
    prompt = get_raw_script_prompt(topic, style=style)

    raw_script = _call_text([
        {"role": "user", "content": f"{prompt}\n\nResearch data:\n{raw_data}"}
    ], temperature=0.9, max_tokens=4096)  # Higher temp for more creative scripts

    return _strip_ending_filler(_extract_numbered_script(_normalize_raw(raw_script)))


def _convert_to_json(raw_script: str, topic: str) -> dict:
    """Convert raw viral script to structured JSON with search terms and image expectations.
    Falls back to Groq on any Ollama failure; retries once on unparsable JSON.
    """
    messages = [
        {"role": "system", "content": JSON_STRUCTURE_PROMPT},
        {"role": "user", "content": f"Raw script:\n{raw_script}\n\nTopic: {topic}\n\nConvert to structured JSON now."}
    ]

    for attempt in range(2):
        raw_json = _call_text(messages, temperature=0.3, max_tokens=8192)
        raw_json = _normalize_raw(raw_json)

        with open(os.path.join(TEMP_DIR, "raw_json_response.txt"), "w") as f:
            f.write(raw_json)

        try:
            return _safe_json_loads(raw_json)
        except json.JSONDecodeError as e:
            if attempt == 0:
                print(f"    [script] JSON parse failed ({e.msg[:80]} at char {e.pos}) - retrying")
                continue
            raise e

    raise json.JSONDecodeError("conversion failed", raw_json, 0)


def build_script(topic: str, raw_data: str, style: dict = None) -> dict:
    """Build script from topic + research data using two-step pipeline.

    style: optional writing-style dict (src/config/writing_styles.py) that
    shapes how the narrator's script sounds for a given voice persona.

    Step 1: Generate raw spoken script using viral script prompt (Cloudflare llama-3.3-70b primary, Groq fallback)
    Step 2: Convert raw script to structured JSON (Cloudflare llama-3.3-70b primary, Groq fallback)
    """
    # Step 1: Generate raw script
    print("    [script] Generating raw script...")
    raw_script = _generate_raw_script(topic, raw_data, style=style)

    # Save raw script for debugging
    with open(os.path.join(TEMP_DIR, "raw_script.txt"), "w") as f:
        f.write(raw_script)

    # Step 2: Convert to structured JSON
    print("    [script] Converting to structured JSON...")
    script = _convert_to_json(raw_script, topic)

    # Catch filler that slipped past the raw-script pass (e.g. added by the
    # JSON structuring model) — strip it from each line's spoken text.
    for line in script.get("lines", []):
        line["text"] = _strip_ending_filler(line.get("text", "")).strip()

    return script