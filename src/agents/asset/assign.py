"""Agentic image->line assignment (was call_groq, tag "assign").

Loop: propose mapping -> validate -> repair once -> return. Every line must end
up with at least one image; anything the model can't place after the repair pass
is left out and the caller's deterministic greedy matcher (still in the asset
agent) fills those. No cloud provider is involved.
"""
from __future__ import annotations

from src.agents.asset.local import complete, extract_json_block, strip_thinking
from src.agents.asset.prompts import assign_prompt, repair_prompt


def _line_expectation(ln: dict) -> str:
    return (
        ln.get("image_features", "")
        or ln.get("image_expectation", "")
        or ln.get("search_term", "")
        or ln.get("text", "")
    )


def _parse_mapping(raw: str, line_ids: list, paths: list) -> dict:
    """Turn a model reply into {line_id(int): [path,...]}, validating indexes."""
    data = extract_json_block(raw if raw else "{}") or {}
    mapping: dict = {}
    for key, val in data.items():
        if not isinstance(val, list):
            continue
        idxs = []
        for i in val:
            if isinstance(i, int):
                idxs.append(i)
            elif isinstance(i, str) and i.strip().isdigit():
                idxs.append(int(i))
        for i in idxs:
            j = min(max(i - 1, 0), len(paths) - 1)
            mapping.setdefault(key, []).append(paths[j])
    # Coerce keys to int and drop anything not covering an actual line.
    valid = {int(ln) for ln in line_ids if str(ln).isdigit()} or set(line_ids)
    out: dict = {}
    for k, v in mapping.items():
        try:
            k = int(k)
        except (TypeError, ValueError):
            continue
        if k in valid:
            out.setdefault(k, [])
            for p in v:
                if p not in out[k]:
                    out[k].append(p)
    return out


def line_expectations(lines: list) -> str:
    return "\n".join(f'line {ln["id"]}: {_line_expectation(ln)}' for ln in lines)


def plan_assignment(images: dict, lines: list, topic: str) -> dict:
    """Local LLM assignment plan {line_id: [paths]} with one repair pass.

    Returns {} when the model produced nothing usable (caller falls back to the
    deterministic greedy matcher). May hold a partial plan if some lines could
    not be filled even after repair.
    """
    if not images or not lines:
        return {}

    paths = list(images.keys())
    catalog = "\n".join(
        f"image {i + 1}: {images[p][0] or images[p][1]}" for i, p in enumerate(paths)
    )
    expectations = line_expectations(lines)
    line_ids = [ln["id"] for ln in lines]

    try:
        raw = strip_thinking(complete(
            assign_prompt(paths, catalog, expectations, line_ids, topic),
            tag="assign", temperature=0.3,
        ))
        mapping = _parse_mapping(raw, line_ids, paths)
    except Exception as e:
        print(f"    [assign] local agent failed ({str(e)[:100]}) - greedy fallback")
        return {}

    missing = [lid for lid in line_ids if not mapping.get(lid)]
    if missing:
        try:
            raw2 = strip_thinking(complete(
                repair_prompt(topic, catalog, expectations, missing),
                tag="as_repair", temperature=0.2,
            ))
            repair = _parse_mapping(raw2, missing, paths)
            for lid, deck in repair.items():
                mapping.setdefault(lid, [])
                for p in deck:
                    if p not in mapping[lid]:
                        mapping[lid].append(p)
        except Exception as e:
            print(f"    [assign] repair pass failed ({str(e)[:100]}) - continuing with partial")

    return mapping