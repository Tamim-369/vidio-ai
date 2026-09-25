"""Central LLM plumbing shared by every stage.

This module is the ONE seam every stage calls: the script stages (scene /
story / script / query), the asset agent, and research all go through
``_local``. Provider routing:

  * default: Groq — uses exactly the 3 keys GROQ_API_KEY / GROQ_API_KEY_SECOND
    / GROQ_API_KEY_THIRD from .env, and the chain only touches Groq when at
    least one of those keys is actually present.
  * set LLM_PROVIDER=ollama in .env -> local Ollama (LOCAL_MODEL).
  * if LLM_PROVIDER=groq but no Groq key exists in .env -> local Ollama
    fallback so the pipeline still runs.

There is NO Gemini/Google anywhere. Groq calls go through
:mod:`src.services.providers` (3-key rotation -> Cloudflare last resort);
Ollama calls use the worker below with a hard wall-clock timeout.
"""
from __future__ import annotations

import os
import queue
import threading

from dotenv import load_dotenv
import ollama

load_dotenv()

# The big local model used by story/script/query/asset stages on this CPU box
# when LLM_PROVIDER=ollama (or no Groq key is configured). .env is loaded
# above so the module-level read below sees the real value.
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma3:4b")

# Provider selection. "groq" is the default; "ollama" opts into the local box.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

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


def _provider() -> str:
    """Resolve the active provider: "groq" (default) or "ollama".

    Groq is used ONLY when a Groq key actually exists in .env — if the user
    asked for groq but no key is present (or deliberately set ollama), the
    pipeline falls back to local Ollama instead of failing.
    """
    if LLM_PROVIDER == "ollama":
        return "ollama"
    if LLM_PROVIDER == "groq":
        from src.services.providers import _groq_keys_present
        if _groq_keys_present():
            return "groq"
        print("  [llm] LLM_PROVIDER=groq but no GROQ_API_KEY* in .env — using local Ollama")
        return "ollama"
    print(f"  [llm] unknown LLM_PROVIDER={LLM_PROVIDER!r} — using local Ollama")
    return "ollama"


def _model_for(tag: str) -> str:
    return _MODELS.get(tag, LOCAL_MODEL)


def _generate(prompt: str, temperature: float, budget_tag: str, model: str | None = None, repeat_penalty: float = 1.1) -> str:
    """One generation call through the central seam.

    Groq (default) when a key exists; local Ollama otherwise. `budget_tag`
    selects the token budget / model routing (NUM_PREDICT above) so each stage
    keeps its existing quotas on both backends.
    """
    if _provider() == "groq":
        from src.services.providers import call_groq
        # No max_tokens here: qwen (primary) caps output tokens per minute and
        # gpt-oss-120b (fallback) is a reasoning model whose small caps are
        # consumed by its `reasoning` field, leaving `content` empty. Let Groq
        # run to completion; token budgets only constrain the local Ollama path
        # (NUM_PREDICT below).
        return call_groq(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            tag=budget_tag,
        )

    # --- local Ollama path ---
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