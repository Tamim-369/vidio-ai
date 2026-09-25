"""Viral topic brainstormer (runs through the central LLM seam).

One grounded call produces a batch of fresh, on-niche topic seeds. The model
answer flows through ``src.agents.common.llm._local`` — the SAME central seam
every other stage uses (Groq by default, local Ollama when LLM_PROVIDER=ollama).
No Gemini/Google here: the previous Google Search grounding is gone; Wikipedia
verification is enforced by the caller (``generator._verify_on_wikipedia``).

Falls back to None (caller decides) if the seam is unreachable.
"""
import json
import re

from src.agents.common.llm import _local
from src.agents.topic.prompts import GEMINI_TOPICS_PROMPT

MAX_TOPICS_PER_CALL = 20

# Prompt text says "search grounding" in one rule; Groq has no web search tool,
# but the caller's Wikipedia verification keeps every seed real anyway.
_GROUNDING_LINE = (
    " (Enforcement note: use only facts you are sure are documented - the caller "
    "verifies each title on Wikipedia before accepting it.)"
)


def _brainstorm_chat(prompt: str):
    """One call through the central LLM seam; returns text or None."""
    try:
        return _local(prompt, temperature=0.9, tag="research")
    except Exception as e:
        print(f"  [topics] brainstorm call failed: {e}")
        return None


def _parse_seeds(text: str) -> list:
    """Extract the JSON list from the reply into {title, angle, summary}."""
    try:
        m = re.search(r"\[.*\]", text, re.S)
        data = json.loads(m.group(0))
    except Exception:
        print("  [topics] Could not parse response")
        return []

    seeds = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        angle = str(item.get("angle", "mystery")).strip()
        if angle not in ("mystery", "scary", "injustice", "heroic"):
            angle = "mystery"
        seeds.append({
            "title": title,
            "angle": angle,
            "summary": str(item.get("summary", "")).strip(),
        })
    return seeds


def brainstorm_topics(count: int = 20, blocklist: list = None) -> list:
    """Return a list of {title, angle, summary} topic seeds.

    Returns an empty list if the central LLM call fails — the caller falls
    back to the other topic sources.
    """
    blocked = blocklist or []
    prompt = GEMINI_TOPICS_PROMPT.format(
        target=min(count, MAX_TOPICS_PER_CALL),
        blocklist="\n".join(f"- {b}" for b in blocked) if blocked else "(none)",
    ) + _GROUNDING_LINE

    response = _brainstorm_chat(prompt)
    return _parse_seeds(response or "") if response else []