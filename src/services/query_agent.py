"""Query Agent: topic-grounded image-search queries for a video script.

One local LLM pass after script generation. Reads the WHOLE script + the
TOPIC, and writes every line's queries WITH THE TOPIC AS GROUND TRUTH - not
the sentence's literal words. Nicknames/vague references are resolved to the
real named subject this topic owns ("Hitler's buzzsaw" -> MG42), era/junk/kind
words are applied, and any query that is not about the topic is replaced with
one that is. Per-line `search_term`/narration text is the only fallback, so
asset fetching never blocks on this agent.

If every line already carries search_queries (the staged lab pipeline fills
these during script generation), those are reused directly - no redundant
model call.
"""
from __future__ import annotations

import re

from src.config.script_lab_prompts import QUERY_BUILDER_PROMPT
from src.services.script_lab.llm import _local
from src.services.script_lab.text import _render

QUERIES_PER_LINE = 3
MAX_LINES = 40


def _strip_thinking_tags(text: str) -> str:
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s*thinking\s*\n", "", text)
    return text.strip()


def _extract_queries(raw: str, line_ids: list) -> dict:
    """Robustly pull {line_id: [query, ...]} out of a local model reply.

    The small local model frequently writes its own escaped-quote scheme
    (``""phrase""`` instead of ``"phrase"``), curly/smart quotes (``\u201c``/
    ``\u201d``), and empty placeholder strings. We normalize quotes and scan
    ``"line": [...]`` buckets with a tolerant pattern, so malformed bits just
    drop out instead of failing the whole pass. Real queries always survive;
    empties are filtered by the caller.
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


def _clean_queries(raw_list: list) -> list:
    out = []
    for q in raw_list:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if q.startswith('"') and q.endswith('"') and len(q) > 2:
            q = q[1:-1].strip()
        if len(q.split()) >= 2:
            out.append(q)
        if len(out) >= QUERIES_PER_LINE:
            break
    return out


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