"""Staged script-production pipeline (the primary script source for main.py).

Pipeline (each stage = one narrow local-model job, run in a fixed order by
`pipeline.run_pipeline`):
  1. facts  - story text -> concrete facts (JSON)
  2. story  - facts -> gripping short spoken script (plain prose, NO JSON)
  3. theme  - prose -> prose in the Arnold/Trump/narrator voice
  4. lines  - prose -> structured JSON lines (beat + tone + text)
  5. queries- per-line image queries (>=2/line, reuse with a gap enforced in code)

The viral/quality principles live in the prompts (my design) and in the
deterministic query-spacing + structural guarantees in this package. No cloud
model is used for writing or for judging - phi4-mini writes, we inspect.

Module layout (one concern per file, stage modules in pipeline order):
    config.py    env knobs (model, timeouts, token budgets)
    llm.py       local Ollama generation plumbing
    text.py      pure text/JSON helpers (tolerant parsing, sentence count)
    numbers.py   number fidelity engine (canonical values + audit)
    facts.py     stage 1 - facts extraction + story angle
    story.py     stage 2 - narrated script (best-of-N scoring)
    theme.py     stage 3 - persona re-voice (number-anchored)
    lines.py     stage 4 - structured JSON lines
    queries.py   stage 5 - image queries (forced >=2, no consecutive repeats)
    pipeline.py  driver - runs the five stages in the single fixed order
    adapter.py   boundary to main.py (build_lab_script, theme_for_style)

Public API:
    run_pipeline(topic, story, theme)   -> raw result dict
    build_lab_script(topic, story, theme) -> normalized script for the video
                                             pipeline (what main.py consumes)
    theme_for_style(style)              -> theme id for a voice style
"""
from src.services.script_lab.adapter import build_lab_script, theme_for_style
from src.services.script_lab.pipeline import run_pipeline

__all__ = ["run_pipeline", "build_lab_script", "theme_for_style"]