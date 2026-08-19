import json
import re
import time
from groq import Groq
import ollama  # <-- added
from src.config.settings import GROQ_API_KEY, GROQ_API_KEY_BACKUP, GROQ_MODEL
from src.config.prompt import get_raw_script_prompt

# Initialize primary client
client = Groq(api_key=GROQ_API_KEY)
backup_client = None
using_backup = False

def _get_groq_client():
    """Get the current Groq client (primary or backup)."""
    global client, backup_client, using_backup
    
    if using_backup and backup_client:
        return backup_client
    return client

def _switch_to_backup():
    """Switch to backup API key if available."""
    global backup_client, using_backup
    
    if GROQ_API_KEY_BACKUP and not using_backup:
        print("    [groq] Switching to backup API key...")
        backup_client = Groq(api_key=GROQ_API_KEY_BACKUP)
        using_backup = True
        return True
    return False

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
      "duration": <3-7 seconds as integer>
    },
    {
      "id": 2,
      "text": "<original text from line 2, without the number prefix>",
      "search_term": "<3-8 word search query>",
      "image_expectation": "<15-20 word visual description>",
      "image_type": "<stock|search>",
      "duration": <3-7 seconds as integer>
    }
  ]
}

Rules:
- Create one object per numbered line from the raw script
- text: The original sentence WITHOUT the number prefix (e.g., "1. " or "2. ")
- search_term: 3-8 words MAX. Pure keywords for image search
- image_expectation: 15-20 words. Describe the visual that matches the drama
- image_type: "search" for specific things, "stock" for generic scenes
- duration: How long it takes to speak (3-7 seconds)
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
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
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


def _call_groq(messages: list, temperature: float = 0.7, model: str = None, max_retries: int = 3) -> str:
    if model is None:
        model = GROQ_MODEL

    for attempt in range(max_retries):
        try:
            current_client = _get_groq_client()
            response = current_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"    [groq] Attempt {attempt + 1}/{max_retries}: {e}")

            # Handle rate limit errors
            if "rate_limit" in error_msg or "413" in error_msg or "tokens" in error_msg:
                # Try switching to backup key on first rate limit
                if attempt == 0 and not using_backup:
                    if _switch_to_backup():
                        print(f"    [groq] Retrying with backup key...")
                        continue  # Retry immediately with backup
                
                # If backup also fails or no backup, wait
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1  # Exponential backoff: 1s, 2s, 4s
                    print(f"    [groq] Rate limit hit, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                raise e
            # For other errors, don't retry
            raise e

    raise Exception(f"Failed after {max_retries} retries")


def _call_ollama(messages: list, temperature: float = 0.3, model: str = "minimax-m3:cloud") -> str:
    """Call Ollama (used for JSON structuring)."""
    response = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature": temperature,
        }
    )
    return response["message"]["content"].strip()


def _generate_raw_script(topic: str, raw_data: str) -> str:
    """Generate raw spoken script using the viral script prompt."""
    prompt = get_raw_script_prompt(topic)

    raw_script = _call_groq([
        {"role": "user", "content": f"{prompt}\n\nResearch data:\n{raw_data}"}
    ], temperature=0.9)  # Higher temp for more creative scripts

    return _normalize_raw(raw_script)


def _convert_to_json(raw_script: str, topic: str) -> dict:
    """Convert raw viral script to structured JSON with search terms and image expectations.
    Now uses Ollama + minimax-m3:cloud instead of Groq.
    """
    raw_json = _call_ollama([
        {"role": "system", "content": JSON_STRUCTURE_PROMPT},
        {"role": "user", "content": f"Raw script:\n{raw_script}\n\nTopic: {topic}\n\nConvert to structured JSON now."}
    ], temperature=0.3)

    raw_json = _normalize_raw(raw_json)
    
    # Save the JSON response for debugging
    with open("./raw_json_response.txt", "w") as f:
        f.write(raw_json)
    
    return _safe_json_loads(raw_json)


def build_script(topic: str, raw_data: str) -> dict:
    """Build script from topic + research data using two-step pipeline.

    Step 1: Generate raw spoken script using viral script prompt (still Groq)
    Step 2: Convert raw script to structured JSON format (now Ollama + minimax-m3:cloud)
    """
    # Step 1: Generate raw script
    print("    [script] Generating raw script...")
    raw_script = _generate_raw_script(topic, raw_data)

    # Save raw script for debugging
    with open("./raw_script.txt", "w") as f:
        f.write(raw_script)

    # Step 2: Convert to structured JSON
    print("    [script] Converting to structured JSON (via Ollama)...")
    script = _convert_to_json(raw_script, topic)

    return script