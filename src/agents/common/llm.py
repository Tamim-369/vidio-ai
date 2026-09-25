"""Local Ollama generation plumbing shared by every stage.

All stages call the model through `_local`. A wedged Ollama HTTP response
(rare but seen on this CPU box) must not stall the whole pipeline forever,
and runaway generation must not be able to write until context limit. The
daemon thread lets a timed-out request be abandoned without blocking
interpreter exit.

Model routing is per-stage, because tokens are the bottleneck on this CPU
box. Stages that need judgment (story write, showrunner verdict) run on the
big model; stages that are short extraction (scene pick) run on the fast
model, which is ~3x quicker and fine for a 300-token JSON brief.
"""
from __future__ import annotations

import os
import queue
import threading

from dotenv import load_dotenv
import ollama

load_dotenv()

# The big local model used by story/script/query/asset stages on this CPU box.
# .env is loaded above so the module-level read below sees the real value.
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma3:4b")

# Token budgets and hard wall-clock timeout for local generation, keyed by
# stage tag (see NUM_PREDICT below). "scene" explicitly maps to the BIG model
# (LOCAL_MODEL): it is the foundation of the entire story, and the fast 1b
# model proved too unreliable there.
STAGE_TIMEOUT_S = int(os.getenv("STAGE_TIMEOUT_S", "300"))
NUM_PREDICT = {
    "scene": 700,       # story pre-pass: material -> ONE incident brief (+facts)
    "story": 1200,
    "eval": 500,        # showrunner: story draft -> verdict + line-level notes
    "json": 1600,
    "research": 2000,
    "kw": 500,          # asset_agent: topic -> keyword phrases
    "assign": 1200,     # asset_agent: image -> line JSON mapping
    "as_repair": 800,   # asset_agent: one self-correction pass
    "queries": 3500,    # query builder stage: topic-grounded per-line queries
}

_MODELS: dict[str, str] = {"scene": LOCAL_MODEL}


def _model_for(tag: str) -> str:
    return _MODELS.get(tag, LOCAL_MODEL)


def _generate(prompt: str, temperature: float, budget_tag: str, model: str | None = None, repeat_penalty: float = 1.1) -> str:
    """Local Ollama generation with a hard wall-clock timeout and token cap."""
    box = queue.Queue(maxsize=1)
    m = model or _model_for(budget_tag)

    def worker():
        try:
            resp = ollama.chat(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": NUM_PREDICT.get(budget_tag, 1000),
                    "repeat_penalty": repeat_penalty,
                },
            )
            box.put(("ok", (resp["message"]["content"] or "").strip()))
        except Exception as e:  # noqa: BLE001
            box.put(("err", e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        kind, val = box.get(timeout=STAGE_TIMEOUT_S)
    except queue.Empty:
        raise RuntimeError(
            f"[local] {m} timed out after {STAGE_TIMEOUT_S}s ({budget_tag})"
        )
    if kind == "err":
        raise val
    return val


def _local(prompt: str, temperature: float = 0.5, tag: str = "story", model: str | None = None, repeat_penalty: float = 1.1) -> str:
    return _generate(prompt, temperature, tag, model, repeat_penalty)