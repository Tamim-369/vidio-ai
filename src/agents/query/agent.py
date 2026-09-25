"""Query agent: topic-grounded image-search queries for a video script.

Two entry points share one prompt + one machinery set:

- ``build_queries(lines, topic)`` - the staged-lab path: attaches per-line
  ``queries`` lists in place and enforces the deterministic guarantees.
- ``build_search_queries(lines, topic)`` - the asset-fetching path: returns a
  ``{line_id: [query, ...]}`` map and attaches ``search_queries`` onto each
  line. Reuses existing ``search_queries`` (which the lab pipeline fills
  during script generation) with no redundant model call.

Both read the WHOLE script + the TOPIC and resolve nicknames to the real
named subject; anything off-topic is replaced.
"""
from __future__ import annotations

from src.agents.common.llm import _local
from src.agents.common.text import _render
from src.agents.query.helpers import (
    _clean_queries,
    _enforce_rule,
    _extract_queries,
    _parse_query_map,
    _strip_thinking_tags,
    _text_of,
)
from src.agents.query.prompts import QUERY_BUILDER_PROMPT

QUERIES_PER_LINE = 3
MAX_LINES = 40


def build_queries(lines: list[dict], topic: str = "") -> list[dict]:
    """One local call -> topic-grounded per-line image queries.

    Works off the TOPIC, not the sentence word-for-word: the prompt teaches
    the model to resolve nicknames/slang via the topic to the real named
    subject, and to drop any query that is not about the topic. Attaches the
    queries as ``ln["queries"]`` and applies the deterministic guarantees.
    """
    script = "\n".join(
        f'{i}. {_text_of(ln.get("text"))}'
        for i, ln in enumerate(lines, 1)
    )
    prompt = _render(QUERY_BUILDER_PROMPT, topic=(topic or "unknown"), script=script)
    out = _local(prompt, temperature=0.2, tag="queries")
    data = _parse_query_map(out)

    for i, ln in enumerate(lines, 1):
        qs = _clean_queries(data.get(i, []))
        ln["queries"] = qs if qs else []
    return _enforce_rule(lines)


def build_search_queries(lines: list, topic: str = "") -> dict:
    """One LLM pass -> {line_id: [query, query, ...]} for every line.

    Also attaches ``line["search_queries"]`` onto the dict for each input line.
    Falls back to the existing ``search_term`` when the model pass fails,
    so asset fetching never blocks on this agent.

    If every line already carries search_queries (the staged lab pipeline fills
    these locally during script generation), those are reused directly - no
    redundant model call. Entries with no useful line text contribute an
    empty list.
    """
    if lines and all(ln.get("search_queries") for ln in lines):
        for ln in lines:
            ln.setdefault("search_queries", [])
        return {ln["id"]: list(ln["search_queries"]) for ln in lines}

    script = "\n".join(
        f'{ln["id"]}. {ln.get("text", "")}'
        for ln in lines[:MAX_LINES]
    )
    user_prompt = _render(
        QUERY_BUILDER_PROMPT,
        topic=(topic or "unknown"),
        script=script,
    )
    try:
        raw = _local(
            user_prompt,
            temperature=0.2,
            tag="queries",
        )
        raw = _strip_thinking_tags(raw)
        data = _extract_queries(raw, [ln["id"] for ln in lines])
        if not data:
            raise ValueError(f"no usable mapping in query-agent reply: {raw[:200]}")
    except Exception as e:
        print(f"    [queries] agent failed ({str(e)[:120]}) - using per-line search_term")
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