import json
import re
from groq import Groq
from src.config.settings import GROQ_API_KEY, GROQ_MODEL, VIDEO_STYLE

client = Groq(api_key=GROQ_API_KEY)

STYLE_PROMPTS = {
    "educational": "You are a scriptwriter for short educational videos. Write in a clear, engaging, fact-driven tone.",
    "motivational": "You are a scriptwriter for motivational short videos. Write in an inspiring, energetic, emotional tone.",
    "ad": "You are a scriptwriter for short product/service ads. Write in a persuasive, punchy, benefit-focused tone.",
    "storytelling": "You are a scriptwriter for short story videos. Write in a narrative, suspenseful, engaging tone.",
    "attraction": (
        "You are a scriptwriter for short viral male self-improvement and attractiveness videos. "
        "Write in a direct, confident, slightly provocative tone that hooks men aged 18-35. "
        "Use specific, visual, data-driven language. Make every line feel like a revelation. "
        "The goal is to educate men on what makes them more physically attractive and how to improve. "
        "Subtly hint that an AI tool can analyze and score their attractiveness. "
        "Never be preachy. Be the cool older brother who gives real advice."
    ),
}

SYSTEM_PROMPT = """
You are given a topic and research data. Your job is to write a short video script (8-12 lines).

Return ONLY valid JSON. No explanation, no markdown, no code blocks. Just raw JSON.

Format:
{{
  "topic": "<topic>",
  "style": "<style>",
  "lines": [
    {{
      "id": 1,
      "text": "<one sentence to be narrated>",
      "search_term": "<descriptive image search query for this sentence>",
      "image_type": "<stock|search>",
      "duration": <estimated seconds as integer>
    }}
  ]
}}

Rules:
- Each line is ONE sentence, narrated by TTS — write it EXACTLY as it should be spoken out loud
- NEVER use a hyphen as a sentence connector or pause (e.g. "attractor-open" or "bulk-women" is wrong). Use a comma or period instead
- Hyphens are ONLY allowed in compound words like "waist-to-hip", "three-day-a-week", or number ranges like "3-4"
- NEVER use double quotes inside text or search_term values — use single quotes instead if needed
- NEVER use a hyphen as a sentence connector or pause in text (e.g. "Hulk-women" or "stamina-women" is wrong). Use a comma or period instead. Hyphens are only allowed in compound words like "waist-to-hip" or number ranges like "3-4".
- search_term must describe what a CAMERA would photograph — think like a photographer, not a summarizer
  - BAD: "subtle effect in the eyes" → matches anything with eyes
  - GOOD: "attractive young man confident direct eye contact close up portrait"
- search_term must ALWAYS include the main topic context so results stay on-topic
- duration should match how long it takes to say the text naturally (3-8 seconds)
- 8 to 12 lines total
- image_type must be "stock" for common/generic visuals (people, nature, gym, city, doctor, etc.) or "search" for specific/niche visuals (specific person, scientific diagram, historical image, specific event, etc.)
"""


def _safe_json_loads(text: str) -> dict:
    """Parse JSON, applying common LLM-output repairs before giving up."""
    # 1. Try as-is.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract the first balanced {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            text = candidate

    # 3. Fix unescaped double quotes inside JSON string values.
    # Matches: "key": "value with "quoted words" inside"
    # Strategy: inside each string value, replace " with ' unless it's the delimiter.
    def fix_inner_quotes(m):
        key = m.group(1)
        value = m.group(2)
        # Replace any unescaped double quote inside the value with single quote
        value = re.sub(r'(?<!\\)"', "'", value)
        return f'"{key}": "{value}"'

    repaired = re.sub(r'"(\w+)":\s*"(.*?)"(?=\s*[,}\]])', fix_inner_quotes, text, flags=re.DOTALL)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 4. Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        col = getattr(e, "colno", 0)
        ln = getattr(e, "lineno", 0)
        lines = repaired.splitlines()
        snippet = lines[ln - 1] if 0 < ln <= len(lines) else "<eof>"
        raise json.JSONDecodeError(
            f"{e.msg} at line {ln} col {col}: {snippet!r}",
            repaired,
            e.pos,
        ) from e


def build_script(topic: str, raw_data: str) -> dict:
    """Call Groq and return structured script JSON."""
    style_instruction = STYLE_PROMPTS.get(VIDEO_STYLE, STYLE_PROMPTS["educational"])

    user_prompt = f"""
Topic: {topic}
Style: {VIDEO_STYLE}

Research data:
{raw_data}

Write the video script now.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": style_instruction + "\n\n" + SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code blocks if model wraps response anyway
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Normalize Unicode punctuation so TTS reads it correctly.
    # Models frequently emit typographic dashes/quotes that TTS engines
    # either skip or read out by name (e.g. "en dash").
    replacements = {
        "\u2011": "-",   # non-breaking hyphen  ‑  -> ASCII hyphen
        "\u2013": "-",   # en dash              –  -> ASCII hyphen
        "\u2014": "-",   # em dash              —  -> ASCII hyphen
        "\u2015": "-",   # horizontal bar       ―  -> ASCII hyphen
        "\u2212": "-",   # minus sign           −  -> ASCII hyphen
        "\u2018": "'",   # left single quote    ‘
        "\u2019": "'",   # right single quote   ’
        "\u201A": "'",   # single low-9 quote   ‚
        "\u201C": '"',   # left double quote    “
        "\u201D": '"',   # right double quote   ”
        "\u201E": '"',   # double low-9 quote   „
        "\u2026": "...", # ellipsis             …
    }
    for src, dst in replacements.items():
        raw = raw.replace(src, dst)

    # Dump the normalized text so debugging matches what TTS actually receives.
    with open("./raw_script.json", "w") as f:
        f.write(raw)

    return _safe_json_loads(raw)
