"""Stage 1 - FACTS: story text -> concrete facts + the story angle.

`extract_facts` turns the raw research text into a list of concrete facts.
`extract_angle` then frame-locks the narration around a single hook/contrast
so later stages don't wander between competing takes.
"""
from __future__ import annotations

from src.config.script_lab_prompts import ANGLE_PROMPT, FACTS_PROMPT
from src.services.script_lab.llm import _local
from src.services.script_lab.text import _loads_json, _render


def _coerce_fact(item) -> dict | None:
    """Normalize one model fact item into a dict with string-ish fields.

    Small models occasionally emit nested arrays, plain strings or bare values
    in the facts list; every downstream stage calls `f.get(...)`, so anything
    non-dict must be coerced here (never left to crash later).
    """
    if isinstance(item, dict):
        fact = item.get("fact") or item.get("name")
        kind = item.get("kind")
        value = item.get("value")
        return {
            "fact": str(fact).strip() if fact is not None else "",
            "kind": str(kind) if kind is not None else "string",
            "value": str(value) if value is not None else None,
        }
    if isinstance(item, str):
        return {"fact": item.strip(), "kind": "string", "value": item if item.strip() else None}
    if item is None:
        return None
    return {"fact": str(item), "kind": "string", "value": str(item)}


def extract_facts(story: str) -> list[dict]:
    out = _local(_render(FACTS_PROMPT, story=story), temperature=0.2, tag="facts")
    data = _loads_json(out)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list):
        raise ValueError(f"facts stage: expected a list, got {type(data).__name__}: {out[:160]!r}")
    if len(data) == 1 and isinstance(data[0], list):  # single extra wrapper level
        data = data[0]
    facts = [
        f for raw in data
        for f in [(_coerce_fact(raw) if isinstance(raw, dict) else
                   _coerce_fact(raw) if isinstance(raw, (str, int, float)) else None)]
        if f and f.get("fact")
    ]
    if not facts:
        raise ValueError(f"facts stage: no usable fact items in: {out[:160]!r}")
    return facts


def _facts_text(facts: list[dict]) -> str:
    lines = []
    for f in facts:
        lines.append(f"- {f.get('fact')}")
    return "\n".join(lines)


def extract_angle(facts: list[dict]) -> dict:
    out = _local(
        _render(ANGLE_PROMPT, facts=_facts_text(facts)),
        temperature=0.3, tag="angle",
    )
    try:
        angle = _loads_json(out)
    except ValueError:
        print("    [angle] unparseable reply - using defaults", flush=True)
        return {}
    if isinstance(angle, list):
        angle = next((a for a in angle if isinstance(a, dict)), {})
    if not isinstance(angle, dict):
        print(f"    [angle] unexpected reply shape {type(angle).__name__} - using defaults", flush=True)
        return {}
    return angle


def _facts_block(facts: list[dict]) -> str:
    return "\n".join(
        f"- {f.get('fact') or f.get('name')} "
        f"[{f.get('kind')}={f.get('value')}]"
        for f in facts
    )