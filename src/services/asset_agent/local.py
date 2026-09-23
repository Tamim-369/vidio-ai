"""Local completion seam + output parsing for the asset agents.

Everything funnels through the staged lab's `_local` (Ollama phi4-mini) so the
rest of the pipeline's tuning (num_predict budget, wall-clock timeout,
repeat_penalty) applies unchanged. Strips thinking tags and digs a JSON block
out of raw output the same way the old Groq path did.
"""
from __future__ import annotations

import json
import re

from src.services.script_lab.llm import _local

_THINKING = re.compile(r"<thinking>.*?</thinking>", flags=re.DOTALL)


def strip_thinking(text: str) -> str:
    text = _THINKING.sub("", text)
    text = re.sub(r"\s*thinking\s*\n", "", text)
    return text.strip()


def complete(prompt: str, tag: str = "local", temperature: float = 0.5) -> str:
    """Local Ollama completion; raises RuntimeError on wedged/timeout model."""
    return _local(prompt, temperature=temperature, tag=tag)


def extract_json_block(text: str) -> dict | None:
    """Pull the first {…} JSON object out of a model reply."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None