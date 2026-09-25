"""Adapter: the single boundary between the lab pipeline and main.py.

Everything outside this package talks to the video pipeline's script shape,
never to the lab's internal dicts. `build_lab_script` runs the staged pipeline
and normalizes it deterministically (no extra LLM calls); `speaking_style_for_style`
maps a voice writing-style dict onto a supported speaking style.
"""
from __future__ import annotations

from src.services.script_lab.lines import _text_of
from src.services.script_lab.pipeline import run_pipeline
from src.services.script_lab.queries import _coerce
from src.utils.file_helpers import dump_artifact


def speaking_style_for_style(style: dict | None) -> str:
    """Resolve a voice writing-style dict to a speaking style id.

    Styles that map onto a persona (arnold/trump/tate) get that speaking style;
    the plain narrator style maps to the neutral speaking style.
    """
    name = ((style or {}).get("name") or "").lower()
    if "arnold" in name:
        return "arnold"
    if "trump" in name:
        return "trump"
    if "tate" in name:
        return "andrew_tate"
    return "narrator"


def build_lab_script(topic: str, story: str, style: str = "") -> dict:
    """Run the staged lab pipeline and normalize its output to the script dict
    shape the rest of the video pipeline expects.

    The lab lines already carry text + beat + tone + per-line image queries.
    Downstream stages (asset_fetcher/tts/assembler) additionally want ids,
    search_term/image_expectation/image_features/image_type/duration/loud and
    search_queries. We fill those deterministically - no extra LLM calls.
    """
    if not style:
        style = "narrator"
    result = run_pipeline(topic, story, style=style)

    lines = []
    for i, ln in enumerate(result["lines"], 1):
        queries = _coerce(ln.get("queries"))
        text = _text_of(ln.get("text"))
        features = " ".join(queries) or text
        beat = _text_of(ln.get("beat")) or ("hook" if i == 1 else "closer" if i == len(result["lines"]) else "setup")
        tone = _text_of(ln.get("tone")) or "normal"
        lines.append({
            "id": i,
            "text": text,
            "beat": beat,
            "tone": tone,
            "search_queries": queries,
            "search_term": queries[0] if queries else text[:60],
            "image_expectation": features[:150],
            "image_features": features[:150],
            "image_type": "stock",
            "duration": 5,
            "loud": tone == "shout",
        })

    script = {
        "topic": topic,
        "lines": lines,
        "story": result["story"],
    }
    dump_artifact("script", script, topic)
    return script