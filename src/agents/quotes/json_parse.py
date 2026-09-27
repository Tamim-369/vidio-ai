"""Tolerant JSON parsing for model output."""

from __future__ import annotations

import json
import re


def _find_bracket(text: str, open_c: str, close_c: str) -> str:
    """Return the first balanced [...] or {...} region of text."""
    i = text.find(open_c)
    if i == -1:
        return ""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == open_c:
            depth += 1
        elif text[j] == close_c:
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return text[i:]


def _loads_json(text: str):
    """Parse JSON from model output.

    Tolerates code fences, surrounding prose and truncation. A truncated array
    is salvaged element by element so a wedged model cannot kill the pipeline.
    """
    if not text:
        raise ValueError("empty model output")
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned)
        return obj
    except Exception:
        pass
    if cleaned.startswith("["):
        salvaged = _salvage_array(cleaned)
        if salvaged is not None and salvaged:
            return salvaged
    for open_c, close_c in (("[", "]"), ("{", "}")):
        chunk = _find_bracket(cleaned, open_c, close_c)
        if chunk:
            try:
                return json.loads(chunk)
            except Exception:
                continue
    raise ValueError(f"no JSON in output: {text[:160]!r}")


def _salvage_array(text: str) -> list | None:
    """Decode the complete elements at the start of a truncated JSON array."""
    dec = json.JSONDecoder()
    idx = 1  # skip '['
    items = []
    while True:
        while idx < len(text) and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(text) or text[idx] == "]":
            break
        try:
            obj, idx = dec.raw_decode(text, idx)
        except Exception:
            break
        items.append(obj)
        while idx < len(text) and text[idx] in " \t\r\n":
            idx += 1
        if idx < len(text) and text[idx] != "," and text[idx] != "]":
            break
    return items if items else None
