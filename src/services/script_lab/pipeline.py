"""Pipeline driver: runs the stages in ONE fixed order.

This is the single place where execution order lives. Every downstream module
(main.py, script_lab_test.py) calls `run_pipeline`; nothing else is allowed to
sequence stages itself. Stages stay decoupled - each one imports only its own
dependencies and communicates through the returned dict.
"""
from __future__ import annotations

import time

from src.services.script_lab.config import LOCAL_MODEL
from src.services.script_lab.story import build_story
from src.services.script_lab.lines import to_lines
from src.services.script_lab.queries import build_queries
from src.utils.file_helpers import dump_artifact

# Source of truth for the ordered stage list (logging + any tooling that needs
# to enumerate the pipeline). The function calls below are what actually run.
STAGES = ["story", "agent", "queries"]


def run_pipeline(topic: str, story: str, style: str = "narrator") -> dict:
    print(f"\n{'='*70}\nPIPELINE: {topic}  (style={style}, model={LOCAL_MODEL})\n{'='*70}")

    t = time.monotonic()
    print("\n[1/3] STORY")
    story_text = build_story(story, style=style, topic=topic)
    print(story_text)
    print(f"     ({time.monotonic() - t:.0f}s)")
    dump_artifact("story", story_text, topic)

    t = time.monotonic()
    print("\n[2/3] SCRIPT AGENT")
    lines = to_lines(story_text)
    for i, ln in enumerate(lines, 1):
        print(f"    {i:2d}. [{ln.get('beat','?'):8}|{ln.get('tone','?'):6}] {ln['text']}")
    print(f"     ({time.monotonic() - t:.0f}s)")
    dump_artifact("lines", lines, topic)

    t = time.monotonic()
    print("\n[3/3] QUERY BUILDER (topic-grounded)")
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