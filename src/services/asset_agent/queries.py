"""Local search-query agent — one query set per narration line (was call_text).

The fallback path preserved: if every line already carries ``search_queries``
(the staged lab fills these locally during script generation) those are reused
and no model call happens at all. Otherwise a compact local prompt builds
JSON {line_id: [4 queries]} and per-line ``search_term``/narration text is the
only fallback — never a cloud provider.
"""
from __future__ import annotations

import re

from src.services.asset_agent.local import complete, extract_json_block, strip_thinking

QUERIES_PER_LINE = 5
MAX_LINES = 40

_PROMPT = """Video topic: "{topic}"

Script lines (write image search queries for each):
{script}

Query rules:
- Query = the REAL name of what the viewer must see + a photo class word.
- Prefer exact named subjects ("F-35 Lightning II", "Port Arthur naval base",
  "USS Nimitz", "Strait of Hormuz"), plus era words when the story is historical
  ("1960s", "Vietnam War era").
- Add minus-junk to push out clipart/toys/renders:
  -render -toy -meme -poster -clipart -illustration -logo
- 2-6 words + exclusions; never a full sentence; no "why/what/how/was".
- Give each line {q} different queries covering DIFFERENT angles (subject,
  place, people/uniforms, artifact/map).
- IMAGES are needed for EVERY line, even bridging lines — reuse the closest real
  subject of the era for those. Never leave a line empty.

Reply ONLY with this JSON, one key per line id, {q} strings per key, no markdown,
no explanations:
{{"1": ["query", "query", ...], "2": [...]}}
"""


def _clean_queries(raw_list: list) -> list:
    out = []
    for q in raw_list:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if q.startswith('"') and q.endswith('"') and len(q) > 2:
            q = q[1:-1].strip()
        if 2 <= len(q.split()) <= 14:
            out.append(q)
        if len(out) >= QUERIES_PER_LINE:
            break
    return out


def build_search_queries(lines: list, topic: str = "") -> dict:
    """{line_id: [query, ...]} for every line; never raises."""
    if lines and all(ln.get("search_queries") for ln in lines):
        for ln in lines:
            ln.setdefault("search_queries", [])
        return {ln["id"]: list(ln["search_queries"]) for ln in lines}

    script = "\n".join(
        f'line {ln["id"]}: "{ln.get("text", "")}"'
        for ln in lines[:MAX_LINES]
    )
    data = {}
    try:
        raw = strip_thinking(complete(
            _PROMPT.format(topic=topic[:400], script=script, q=QUERIES_PER_LINE),
            tag="queries",
            temperature=0.2,
        ))
        data = extract_json_block(raw) or {}
    except Exception as e:
        print(f"    [queries] local agent failed ({str(e)[:100]}) - using per-line search_term")
        data = {}

    result = {}
    for ln in lines:
        lid = ln["id"]
        qs = _clean_queries(data.get(str(lid), []) or data.get(lid, []))
        if not qs and ln.get("search_term"):
            qs = [ln["search_term"]]
        if not qs:
            qs = [ln.get("text", "")[:60]]
        result[lid] = qs[:QUERIES_PER_LINE]
        ln["search_queries"] = result[lid]
    return result