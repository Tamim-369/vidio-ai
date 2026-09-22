"""Stage 1 - FACTS: story text -> concrete facts + the story angle.

`extract_facts` turns the raw research text into a list of concrete facts.
`extract_angle` then frame-locks the narration around a single hook/contrast
so later stages don't wander between competing takes.
"""
from __future__ import annotations

from src.config.script_lab_prompts import ANGLE_PROMPT, FACTS_PROMPT
from src.services.script_lab.llm import _local
from src.services.script_lab.text import _loads_json, _render


def extract_facts(story: str) -> list[dict]:
    out = _local(_render(FACTS_PROMPT, story=story), temperature=0.2, tag="facts")
    data = _loads_json(out)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list):
        raise ValueError(f"facts stage: expected a list, got {type(data).__name__}: {out[:160]!r}")
    return data


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
    angle = _loads_json(out)
    if isinstance(angle, list):
        angle = next((a for a in angle if isinstance(a, dict)), {})
    if not isinstance(angle, dict):
        raise ValueError(f"angle stage: expected a dict: {out[:160]!r}")
    return angle


def _facts_block(facts: list[dict]) -> str:
    return "\n".join(
        f"- {f.get('fact') or f.get('name')} "
        f"[{f.get('kind')}={f.get('value')}]"
        for f in facts
    )