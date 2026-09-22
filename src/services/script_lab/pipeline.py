"""Pipeline driver: runs the five stages in ONE fixed order.

This is the single place where execution order lives. Every downstream module
(main.py, script_lab_test.py) calls `run_pipeline`; nothing else is allowed to
sequence stages itself. Stages stay decoupled - each one imports only its own
dependencies and communicates through the returned dict.
"""
from __future__ import annotations

import time

from src.services.script_lab.config import LOCAL_MODEL
from src.services.script_lab.facts import extract_facts
from src.services.script_lab.lines import to_lines
from src.services.script_lab.queries import generate_queries
from src.services.script_lab.story import write_story
from src.services.script_lab.theme import apply_theme

# Source of truth for the ordered stage list (logging + any tooling that needs
# to enumerate the pipeline). The function calls below are what actually run.
STAGES = ["facts", "story", "theme", "lines", "queries"]


def run_pipeline(topic: str, story: str, theme: str = "arnold") -> dict:
    print(f"\n{'='*70}\nPIPELINE: {topic}  (theme={theme}, model={LOCAL_MODEL})\n{'='*70}")

    t = time.monotonic()
    print("\n[1/5] FACTS")
    facts = extract_facts(story)
    for f in facts:
        print(f"    - {f.get('fact')}  [{f.get('kind')}={f.get('value')}]")
    if not facts:
        raise RuntimeError("facts stage returned nothing - cannot continue")
    print(f"     ({time.monotonic() - t:.0f}s)")

    t = time.monotonic()
    print("\n[2/5] STORY")
    story_prose = write_story(topic, facts)
    print(story_prose)
    print(f"     ({time.monotonic() - t:.0f}s)")

    t = time.monotonic()
    print(f"\n[3/5] THEME ({theme})")
    themed = apply_theme(story_prose, theme, facts=facts)
    print(themed)
    print(f"     ({time.monotonic() - t:.0f}s)")

    t = time.monotonic()
    print("\n[4/5] JSON LINES")
    lines = to_lines(themed)
    for i, ln in enumerate(lines, 1):
        print(f"    {i:2d}. [{ln.get('beat','?'):8}|{ln.get('tone','?'):6}] {ln['text']}")
    print(f"     ({time.monotonic() - t:.0f}s)")

    t = time.monotonic()
    print("\n[5/5] IMAGE QUERIES")
    lines = generate_queries(topic, story, lines)
    for i, ln in enumerate(lines, 1):
        print(f"    {i:2d}. {ln['text']}")
        for q in ln["queries"]:
            print(f"         - {q}")
    print(f"     ({time.monotonic() - t:.0f}s)")

    return {
        "facts": facts,
        "story": story_prose,
        "themed": themed,
        "lines": lines,
    }