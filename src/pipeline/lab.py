"""Script pipeline: the orchestrator for the staged script agents.

This is the single public boundary the video pipeline talks to. Everything
runs strictly in the fixed order here; no other module is allowed to sequence
the stages itself. Agents stay decoupled - each imports only its own
dependencies and communicates through the returned dict.

Stages (src/agents/):
  1. scene    - pick ONE dramatic incident from the raw source
  2. story    - dramatize that incident in the narrator's speaking style,
                gated by the showrunner verdict loop + deterministic fact gate
  3. script   - story -> structured JSON lines (beat + tone + text)
  4. queries  - topic-grounded image-search queries per line

Public API:
    run_pipeline(topic, story, style)          -> raw result dict
    build_lab_script(topic, story, style)      -> normalized script dict for
                                                 the video pipeline (what
                                                 main.py consumes)
    speaking_style_for_style(style)            -> speaking style id
"""
from __future__ import annotations

import time

from src.agents.script.agent import to_lines
from src.agents.script.helpers import _text_of
from src.agents.query.agent import build_queries
from src.agents.query.helpers import _coerce
from src.agents.scene.agent import pick_scene
from src.agents.story.agent import build_story
from src.agents.common.llm import LOCAL_MODEL
from src.utils.file_helpers import dump_artifact

# Source of truth for the ordered stage list (logging + any tooling that needs
# to enumerate the pipeline). The function calls below are what actually run.
STAGES = ["scene", "story", "agent", "queries"]


def run_pipeline(topic: str, story: str, style: str = "narrator") -> dict:
    print(f"\n{'='*70}\nPIPELINE: {topic}  (style={style}, story={LOCAL_MODEL})\n{'='*70}")

    t = time.monotonic()
    print("\n[1/4] PICK ONE SCENE")
    scene = pick_scene(story)
    print(f"     incident: {scene.get('incident', '')[:120]}")
    print(f"     person:   {scene.get('person', '')[:80]}")
    print(f"     stakes:   {scene.get('stakes', '')[:80]}")
    print(f"     detail:   {scene.get('detail', '')[:80]}")
    print(f"     ({time.monotonic() - t:.0f}s)")
    dump_artifact("scene", scene, topic)

    t = time.monotonic()
    print("\n[2/4] STORY")
    story_text = build_story(story, style=style, topic=topic, scene=scene)
    print(story_text)
    print(f"     ({time.monotonic() - t:.0f}s)")
    dump_artifact("story", story_text, topic)

    t = time.monotonic()
    print("\n[3/4] SCRIPT AGENT")
    lines = to_lines(story_text)
    for i, ln in enumerate(lines, 1):
        print(f"    {i:2d}. [{ln.get('beat','?'):8}|{ln.get('tone','?'):6}] {ln['text']}")
    print(f"     ({time.monotonic() - t:.0f}s)")
    dump_artifact("lines", lines, topic)

    t = time.monotonic()
    print("\n[4/4] QUERY BUILDER (topic-grounded)")
    lines = build_queries(lines, topic=topic)
    for i, ln in enumerate(lines, 1):
        for q in ln.get("queries", []):
            print(f"    {i:2d}. - {q}")
    print(f"     ({time.monotonic() - t:.0f}s)")
    dump_artifact("queries", lines, topic)

    return {
        "story": story_text,
        "lines": lines,
    }


def speaking_style_for_style(style: dict | None) -> str:
    """Resolve a voice writing-style dict to a speaking style id.

    The main branch has a single speaking style (the neutral narrator);
    persona styles (arnold/trump/tate) live on the wizdom branch.
    """
    return "narrator"


def build_lab_script(topic: str, story: str, style: str = "") -> dict:
    """Run the staged pipeline and normalize its output to the script dict
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