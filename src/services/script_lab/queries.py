"""Stage 5 - QUERIES: per-line image queries, >=2 distinct per line.

The model proposes queries; `_enforce_rule` then makes deterministic
guarantees we never trust the model for: every line has >= 2 DISTINCT queries,
and the same image is never used on two CONSECUTIVE lines (a different image
must appear between two uses of the same one).
"""
from __future__ import annotations

import json

from src.config.script_lab_prompts import QUERY_PROMPT
from src.services.script_lab.llm import _local
from src.services.script_lab.text import _loads_json, _render

_FALLBACK_QUERIES = [
    "ww2 soldiers marching formation",
    "tanks rolling across field",
    "old black and white war photo",
    "military maps and documents close up",
    "soldiers standing in rain dark",
    "searchlights night sky battlefield",
    "warship bow waves ocean",
    "soldier holding vintage rifle",
]


def _coerce(item) -> list[str]:
    """Flatten whatever the model returned into a list of query strings.

    Small models occasionally emit nested arrays (`"queries": [["a"], "b"]`),
    dicts (`{"text": "..."}`) or numbers. Every element is flattened to its
    string fragments so `q.strip()` downstream can never hit a list.
    """
    if isinstance(item, str):
        return [item]
    if isinstance(item, list):
        out = []
        for x in item:
            out.extend(_coerce(x))
        return out
    if isinstance(item, dict):
        for k in ("text", "query", "q", "queries"):
            if k in item:
                return _coerce(item[k])
        return []
    if item is None:
        return []
    return [str(item)]


def _distinct(queries) -> list[str]:
    seen, out = set(), []
    for q in _coerce(queries):
        for k in q.splitlines():
            k = k.strip()
            if k and k.lower() not in seen:
                seen.add(k.lower())
                out.append(k)
    return out


def _replace_dupes(ahead: list[str], prev: set[str]) -> list[str]:
    """Replace any query repeated on the immediate previous line."""
    pool = [q for q in _FALLBACK_QUERIES if q not in prev and q not in ahead]
    result = []
    for i, q in enumerate(ahead):
        if q in prev:
            key = q.lower()
            repl = next((p for p in pool if p not in result and p not in prev), None)
            if repl is not None:
                pool.remove(repl)
                result.append(repl)
                continue
        result.append(q)
    return result


def _enforce_rule(lines: list[dict]) -> list[dict]:
    """Deterministic guarantees (the part we never trust the model for):
    - every line has >= 2 DISTINCT queries
    - the same image is never used on two CONSECUTIVE lines (a different
      image must appear between two uses of the same one)
    """
    prev: set[str] = set()
    for ln in lines:
        ln["queries"] = _distinct(ln.get("queries") or [])
        i = 0
        while len(ln["queries"]) < 2 and i < len(_FALLBACK_QUERIES) * 2:
            q = _FALLBACK_QUERIES[i % len(_FALLBACK_QUERIES)]
            if q not in ln["queries"] and q not in prev:
                ln["queries"].append(q)
            i += 1
        ln["queries"] = _replace_dupes(ln["queries"], prev)
        prev = set(ln["queries"])
    return lines


def generate_queries(topic: str, story: str, lines: list[dict]) -> list[dict]:
    lines_json = json.dumps([{"line": i, "text": ln["text"]} for i, ln in enumerate(lines, 1)])
    out = _local(
        _render(QUERY_PROMPT, topic=topic, story=story, lines_json=lines_json),
        temperature=0.3, tag="query",
    )
    raw = _loads_json(out)
    if isinstance(raw, dict):
        raw = raw.get("lines") or raw.get("queries") or [raw]
    if not isinstance(raw, list):
        raise ValueError("query stage: expected a list")
    qmap: dict = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            line_no = int(item.get("line"))
        except (TypeError, ValueError):
            continue
        qs = item.get("queries")
        if qs:
            qmap[line_no] = qs
    for i, ln in enumerate(lines, 1):
        # Coerce BEFORE the list() wrap: `list({...})` would yield dict keys.
        ln["queries"] = _distinct(qmap.get(i, []) or ln.get("queries", []))
    return _enforce_rule(lines)