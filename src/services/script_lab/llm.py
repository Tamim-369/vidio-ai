"""Local Ollama generation plumbing shared by every stage.

All stages call the model through `_local`. A wedged Ollama HTTP response
(rare but seen on this CPU box) must not stall the whole pipeline forever,
and runaway generation must not be able to write until context limit. The
daemon thread lets a timed-out request be abandoned without blocking
interpreter exit.
"""
from __future__ import annotations

import queue
import threading

import ollama

from src.services.script_lab.config import LOCAL_MODEL, NUM_PREDICT, STAGE_TIMEOUT_S


def _generate(prompt: str, temperature: float, budget_tag: str, repeat_penalty: float = 1.1) -> str:
    """Local Ollama generation with a hard wall-clock timeout and token cap."""
    box = queue.Queue(maxsize=1)

    def worker():
        try:
            resp = ollama.chat(
                model=LOCAL_MODEL,
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
            f"[local] {LOCAL_MODEL} timed out after {STAGE_TIMEOUT_S}s ({budget_tag})"
        )
    if kind == "err":
        raise val
    return val


def _local(prompt: str, temperature: float = 0.5, tag: str = "story", repeat_penalty: float = 1.1) -> str:
    return _generate(prompt, temperature, tag, repeat_penalty)