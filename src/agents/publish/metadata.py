"""Title/description/tag generation for a rendered video.

Asks the LLM for metadata, verifies it, and repairs it once if it fails the
checks, falling back to a deterministic title so a publish never dies on a bad
model response. Pure text work -- no YouTube client, no network.
"""

import json

from src.agents.completion.llm import call_text
from src.agents.publish.prompts import (
    get_metadata_prompt,
    get_metadata_verifier_prompt,
    get_metadata_fix_prompt,
)


def _format_script(script: dict) -> str:
    """Flatten the script lines into readable narration text."""
    topic = script.get("topic", "")
    lines = script.get("lines", [])
    body = "\n".join(f"{i + 1}. {l.get('text', '')}" for i, l in enumerate(lines))
    return f"Topic: {topic}\n\n{body}"


def _fallback_metadata(topic: str, script_text: str) -> dict:
    """Deterministic metadata used when the LLM call fails."""
    title_words = topic.split()
    keyword = " ".join(title_words[:4]).rstrip(".")
    title = f"{keyword}: the dark truth (True Story)"
    title = title[:65] if len(title) > 65 else title

    hook = f"Inside the shocking story of {topic}. What really happened is worse than you think."
    covered = "\n".join(
        f"- {l.strip()}" for l in script_text.splitlines() if l.strip()
    )
    description = (
        f"{hook}\n\n"
        f"In this video we break down {topic} from start to finish.\n\n"
        f"{covered}\n\n"
        "Subscribe for more dark histories and untold stories."
    )
    tags = [
        topic.lower().rstrip("."),
        "history",
        "documentary",
        "dark history",
        "shorts",
        "untold story",
    ]
    return {"title": title, "description": description, "tags": tags}


def _extract_json(raw: str) -> dict:
    """Pull the first {...} JSON object out of an LLM response."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _clean_metadata(meta: dict) -> dict:
    """Validate and normalize a metadata dict; empty values fall back to fallback."""
    title = (meta.get("title") or "").strip()
    description = (meta.get("description") or "").strip()
    tags = meta.get("tags") or []

    if not title or not description or len(title) > 100 or len(description) < 50:
        return {}

    return {
        "title": title,
        "title_alternates": [t for t in (meta.get("title_alternates") or []) if isinstance(t, str)][:3],
        "description": description,
        "tags": [str(t).lower().replace(" ", "-") for t in tags if str(t).strip()][:15],
    }


def generate_metadata(topic: str, script: dict) -> dict:
    """Generate title, description, and tags using minimax-m3:cloud (Ollama).

    Two-pass with a verifier:
      1. The generator is told to think through title/description tactics first.
      2. A verifier (same model) audits the result against every rule and returns
         a PASS/FAIL verdict. On FAIL it is fed the verifier's feedback and
         rewrites, up to MAX_VERIFY_ROUNDS.
    Falls back to deterministic metadata if the LLM calls fail.
    """
    MAX_VERIFY_ROUNDS = 2
    script_text = _format_script(script)

    def _call(prompt: str, temperature: float = 0.6) -> dict:
        try:
            raw = call_text(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return _extract_json(raw)
        except Exception as e:
            print(f"    [youtube] LLM call failed: {e}")
            return {}

    # Pass 1: generate (think-first prompt).
    meta = _call(get_metadata_prompt(topic, script_text), temperature=0.7)
    if not meta:
        print("    [youtube] Metadata generation failed, using fallback")
        return _fallback_metadata(topic, script_text)

    # Pass 2+: verify, and rewrite on FAIL until it passes.
    for round_no in range(1, MAX_VERIFY_ROUNDS + 1):
        verifier = _call(
            get_metadata_verifier_prompt(topic, script_text, json.dumps(meta, indent=2)),
            temperature=0.2,
        )
        verdict = (verifier.get("verdict") or "").strip().upper()
        print(
            f"    [youtube] Verifier round {round_no}: {verdict}"
            f" (title {verifier.get('title_score')}/100,"
            f" desc {verifier.get('description_score')}/100,"
            f" emotion: {verifier.get('emotion_triggered')})"
        )

        if verdict == "PASS":
            # Prefer the verifier's fixed title if it supplied a stronger one.
            fixed_title = (verifier.get("fixed_title") or "").strip()
            if fixed_title and _clean_metadata({"title": fixed_title, "description": meta.get("description", ""), "tags": meta.get("tags", [])}):
                meta["title"] = fixed_title
            break

        # FAIL: rewrite with the verifier's feedback.
        meta = _call(
            get_metadata_fix_prompt(topic, script_text, json.dumps(meta, indent=2), json.dumps(verifier, indent=2)),
            temperature=0.7,
        )
        if not meta:
            break

    cleaned = _clean_metadata(meta)
    if not cleaned:
        print("    [youtube] LLM metadata invalid after verification, using fallback")
        return _fallback_metadata(topic, script_text)

    return cleaned
