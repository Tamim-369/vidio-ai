"""Query agent - deterministic helpers (no model calls).

Tolerant extraction of ``{line_no: [query, ...]}`` from small-model replies,
plus the deterministic guarantees `_enforce_rule` gives (>=2 distinct per
line, no consecutive repeats) - the parts the pipeline never trusts the model
for. Both entry points (:func:`build_queries` and
:func:`build_search_queries` in the agent module) share this machinery.
"""
from __future__ import annotations

import re

_QUERIES_PER_LINE = 3

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
            repl = next((p for p in pool if p not in result and p not in prev), None)
            if repl is not None:
                pool.remove(repl)
                result.append(repl)
                continue
        result.append(q)
    return result


def _enforce_rule(lines: list[dict]) -> list[dict]:
    """Deterministic guarantees (the part we never trust the model for):
    - every line has _QUERIES_PER_LINE distinct queries (model leftovers are
      kept and the fallback pool tops the line up to 3)
    - the same query is never used on two CONSECUTIVE lines (a different
      query must appear between two uses of the same one)
    """
    prev: set[str] = set()
    for ln in lines:
        ln["queries"] = _distinct(ln.get("queries") or [])
        i = 0
        while len(ln["queries"]) < _QUERIES_PER_LINE and i < len(_FALLBACK_QUERIES) * 2:
            q = _FALLBACK_QUERIES[i % len(_FALLBACK_QUERIES)]
            if q not in ln["queries"] and q not in prev:
                ln["queries"].append(q)
            i += 1
        ln["queries"] = _replace_dupes(ln["queries"], prev)
        prev = set(ln["queries"])
    return lines


def _text_of(v) -> str:
    """Flatten any model-provided field to a trimmed string (never crashes)."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return " ".join(p for p in (_text_of(x) for x in v) if p).strip()
    if isinstance(v, dict):
        return _text_of(next((v[k] for k in ("text", "value", "beat") if k in v), ""))
    return str(v).strip()


def _parse_query_map(out: str) -> dict[int, list[str]]:
    """Tolerantly pull {line_no: [query, ...]} out of a local model reply.

    The small local model frequently writes its own escaped quote scheme
    (``""phrase""``), curly/smart quotes (``\u201c``/``\u201d``), and empty
    placeholders. Collapse quote runs, normalize curly quotes, and scan
    ``"N": [...]`` buckets with a tolerant pattern, so malformed bits drop out
    instead of failing the whole stage. Also accepts a bare list of lists in
    order.
    """
    block = re.search(r"\{.*\}", out, flags=re.DOTALL)
    if not block:
        return {}
    block = block.group(0).replace('\\"', '"')
    block = block.replace("\u201c", '"').replace("\u201d", '"')
    block = re.sub(r'"+', '"', block)

    result: dict[int, list[str]] = {}
    for m in re.finditer(r'"(\d+)"\s*:\s*\[(.*?)\]', block, flags=re.DOTALL):
        arr = m.group(2)
        tok = re.findall(r'"([^"]*)"', arr)
        qs = [t.strip().strip(",") for t in tok if t.strip() and t.strip() != ","]
        result[int(m.group(1))] = qs
    return result


def _strip_thinking_tags(text: str) -> str:
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s*thinking\s*\n", "", text)
    return text.strip()


def _extract_queries(raw: str, line_ids: list) -> dict:
    """Robustly pull {line_id: [query, ...]} out of a local model reply.

    Stricter sibling of :func:`_parse_query_map`: only line ids actually in
    the script are accepted (a hallucinated extra bucket is ignored), and the
    result keys are the line ids as given by the caller.
    """
    block = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not block:
        return {}
    block = block.group(0).replace('\\"', '"')
    block = block.replace("\u201c", '"').replace("\u201d", '"')
    block = re.sub(r'"+', '"', block)

    out: dict = {}
    valid = {int(lid) for lid in line_ids if str(lid).isdigit()} or set(line_ids)
    for m in re.finditer(r'"(\d+)"\s*:\s*\[(.*?)\]', block, flags=re.DOTALL):
        lid = int(m.group(1))
        if lid not in valid:
            continue
        arr = m.group(2)
        tok = re.findall(r'"([^"]*)"', arr)
        qs = [t.strip().strip(",") for t in tok if t.strip() and t.strip() != ","]
        out[str(lid)] = qs
    return out


def _clean_queries(raw_list: list, cap: int | None = None) -> list[str]:
    out = []
    for q in raw_list:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if q.startswith('"') and q.endswith('"') and len(q) > 2:
            q = q[1:-1].strip()
        if len(q.split()) >= 2:
            out.append(q)
        if cap and len(out) >= cap:
            break
    return out