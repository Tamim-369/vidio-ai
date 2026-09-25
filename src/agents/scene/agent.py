"""Scene agent: turn raw source material into ONE dramatic incident brief.

Entry point: ``pick_scene(story, avoid)``. This is the narrowest pass in the
pipeline - deliberately so. The writer below receives only this brief, never
the full article, which is what forces non-chronological, dramatized
storytelling.
"""
from __future__ import annotations

from src.agents.common.llm import _local
from src.agents.common.text import _loads_json, _render, _scalar
from src.agents.scene.prompts import SCENE_SELECT_PROMPT


def pick_scene(story: str, avoid: list[str] | None = None) -> dict:
    """Pass 1: read the material, return ONE dramatic incident as a JSON brief.

    Returns a dict with keys incident/person/stakes/detail/outcome/who_else/
    numbers. Empty fields are tolerated; the writer still gets a usable prompt
    either way. A crash or a non-JSON reply falls back to the whole (trimmed)
    material as the incident.

    ``avoid`` lists incidents the showrunner already rejected as unmakeable;
    they are surfaced to the scene picker so it hands the writer a different
    moment instead of the same dead end.
    """
    prompt = _render(SCENE_SELECT_PROMPT, story=story[:16000])
    if avoid:
        skips = "\n".join(f"- {a[:200]}" for a in avoid if a)
        prompt += (
            "\n\nThese incidents were already tried and REJECTED as unmakeable."
            " Pick a DIFFERENT incident - not any of these:\n"
            f"{skips}"
        )
    out = _local(prompt, temperature=0.4, tag="scene")
    try:
        brief = _loads_json(out or "")
    except ValueError:
        brief = None
    if isinstance(brief, dict):
        return {
            "incident": _scalar(brief.get("incident")) or "",
            "person": _scalar(brief.get("person")) or "",
            "stakes": _scalar(brief.get("stakes")) or "",
            "detail": _scalar(brief.get("detail")) or "",
            "outcome": _scalar(brief.get("outcome")) or "",
            "who_else": _scalar(brief.get("who_else")) or "",
            "numbers": _scalar(brief.get("numbers")) or "",
            "from_scene": True,
        }
    # Fallback: writer gets the (trimmed) raw material as the incident.
    return {
        "incident": story[:4000],
        "person": "",
        "stakes": "",
        "detail": "",
        "outcome": "",
        "who_else": "",
        "numbers": "",
        "from_scene": False,
    }