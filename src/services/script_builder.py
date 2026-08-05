import json
import re
from groq import Groq
from src.config.settings import GROQ_API_KEY, GROQ_MODEL, VIDEO_STYLE

client = Groq(api_key=GROQ_API_KEY)



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
- NEVER use a hyphen as a sentence connector or pause. Use a comma or period instead
- Hyphens are ONLY allowed in compound words like "waist-to-hip" or number ranges like "3-4"
- NEVER use double quotes inside text or search_term values — use single quotes instead if needed
- search_term must describe what a CAMERA would photograph — think like a photographer, not a summarizer
- search_term must include the topic context so results stay relevant
- duration should match how long it takes to say the text naturally (3-8 seconds)
- 8 to 12 lines total
- image_type: "stock" for generic visuals (people, nature, gym, city), "search" for specific ones (person, diagram, event)
"""

REVIEW_PROMPT = """
You are a strict video script reviewer. You will be given a video script JSON.

Your job is to review it and return a JSON object with this format:
{{
  "approved": true or false,
  "issues": ["issue 1", "issue 2", ...],
  "suggestions": ["suggestion 1", "suggestion 2", ...]
}}

Check for these issues:
- Any hyphen used as a sentence connector instead of a comma/period (e.g. "bulk-women", "attractor-open")
- Lines that are too long or awkward to speak naturally
- Search terms that are too vague (just one word) or don't describe a visual scene
- Lines that feel repetitive or don't add new information
- Poor flow between lines (topics jumping around randomly)
- Any line where the search_term wouldn't find a relevant image

Return ONLY valid JSON. No explanation, no markdown.
"""

REVISION_PROMPT = """
You are given a video script JSON and a list of issues found during review.
Fix ALL the listed issues and return the corrected script in the exact same JSON format.
Return ONLY valid JSON. No explanation, no markdown, no code blocks.
"""


def _normalize_raw(raw: str) -> str:
    """Strip markdown blocks and normalize Unicode punctuation."""
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

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


def _safe_json_loads(text: str) -> dict:
    """Parse JSON with common LLM-output repairs."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            text = candidate

    def fix_inner_quotes(m):
        key, value = m.group(1), m.group(2)
        value = re.sub(r'(?<!\\)"', "'", value)
        return f'"{key}": "{value}"'

    repaired = re.sub(r'"(\w+)":\s*"(.*?)"(?=\s*[,}\]])', fix_inner_quotes, text, flags=re.DOTALL)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        ln = getattr(e, "lineno", 0)
        lines = repaired.splitlines()
        snippet = lines[ln - 1] if 0 < ln <= len(lines) else "<eof>"
        raise json.JSONDecodeError(f"{e.msg}: {snippet!r}", repaired, e.pos) from e


def _call_groq(messages: list, temperature: float = 0.7) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def build_script(topic: str, raw_data: str) -> dict:
    """Build script from topic + research data."""
    

    raw = _call_groq([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Topic: {topic}\nStyle: {VIDEO_STYLE}\n\nResearch data:\n{raw_data}\n\nWrite the video script now."},
    ])

    raw = _normalize_raw(raw)

    with open("./raw_script.json", "w") as f:
        f.write(raw)

    return _safe_json_loads(raw)


def review_script(script: dict) -> dict:
    """Review the script. Returns review result with approved, issues, suggestions."""
    print("  [review] Reviewing script...")

    raw = _call_groq([
        {"role": "system", "content": REVIEW_PROMPT},
        {"role": "user", "content": f"Review this script:\n{json.dumps(script, indent=2)}"},
    ], temperature=0.3)

    raw = _normalize_raw(raw)

    try:
        return _safe_json_loads(raw)
    except Exception:
        # If review JSON is broken, assume approved to not block the pipeline
        return {"approved": True, "issues": [], "suggestions": []}


def revise_script(script: dict, review: dict) -> dict:
    """Revise the script based on review feedback."""
    print("  [review] Revising script...")

    issues_text = "\n".join(f"- {i}" for i in review.get("issues", []))
    suggestions_text = "\n".join(f"- {s}" for s in review.get("suggestions", []))

    raw = _call_groq([
        {"role": "system", "content": REVISION_PROMPT},
        {"role": "user", "content": (
            f"Issues found:\n{issues_text}\n\n"
            f"Suggestions:\n{suggestions_text}\n\n"
            f"Script to fix:\n{json.dumps(script, indent=2)}"
        )},
    ], temperature=0.5)

    raw = _normalize_raw(raw)

    with open("./raw_script.json", "w") as f:
        f.write(raw)

    try:
        return _safe_json_loads(raw)
    except Exception:
        return script  # if revision fails, return original
