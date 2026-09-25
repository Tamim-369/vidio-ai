"""Staged script-production pipeline (the primary script source for main.py).

Pipeline (each stage = one narrow local-model job, run in a fixed order by
`pipeline.run_pipeline`):
  1. story   - source text -> short, detailed, easy-to-follow story, written
               directly in the narrator's speaking style in ONE pass
  2. agent   - story -> structured JSON lines (beat + tone + text)
  3. queries - topic-grounded image-search queries per line (the TOPIC is the
               ground truth: nicknames resolve to the real subject, off-topic
               queries are discarded)

The viral/quality principles live in the prompts and the character speaking
styles (my design) plus the deterministic query-spacing + structural
guarantees in this package. No cloud model is used for writing or for judging -
phi4-mini writes, we inspect.

Module layout (one concern per file, stage modules in pipeline order):
    config.py    env knobs (model, timeouts, token budgets)
    llm.py       local Ollama generation plumbing
    text.py      pure text/JSON helpers (tolerant parsing, sentence count)
    story.py     stage 1 - one-pass story in the character's speaking style
    lines.py     stage 2 - structured JSON lines (one agent call)
    queries.py   stage 3 - topic-grounded image queries + deterministic
                 guarantees (>=2, no consecutive repeats)
    pipeline.py  driver - runs the stages in the single fixed order
    adapter.py   boundary to main.py (build_lab_script, speaking_style_for_style)

Public API:
    run_pipeline(topic, story, style)   -> raw result dict
    build_lab_script(topic, story, style) -> normalized script for the video
                                             pipeline (what main.py consumes)
    speaking_style_for_style(style)     -> speaking style id for a voice style
"""
from src.services.script_lab.adapter import build_lab_script, speaking_style_for_style
from src.services.script_lab.pipeline import run_pipeline

__all__ = ["run_pipeline", "build_lab_script", "speaking_style_for_style"]