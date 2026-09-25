"""Single seam for LLM calls (Groq + Cloudflare).

Owns the provider lifecycle and fallback chain. The primary pathway is:

    call_text()    -> Groq (3 rotating keys) -> Cloudflare (last resort).
    call_groq()    -> Groq (3 rotating keys).

Groq keys rotate across GROQ_API_KEY / GROQ_API_KEY_SECOND / GROQ_API_KEY_THIRD
on retriable or hard failures, and repeated calls pick up where the last one
succeeded, so a dead key is never re-hit first. Cloudflare is demoted to a
final safety net since it rate-limits (429) constantly. There is intentionally
NO Gemini/Google in the chain: the project models text with Groq (default) or
local Ollama (see ``src/agents/common/llm.py``). No service builds its own
client or retry loop: use ``call_text`` / ``call_groq`` and forget the plumbing.

Note: there is NO vision seam anymore. Image verification was removed from the
pipeline — image selection is driven by the per-line query agent, so model
description/verification calls (and their Groq/Gemini/Ollama vision variants)
are gone.
"""
import os
import time

import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- Cloud LLM provider credentials, read directly here (the only consumer).
# Cloudflare Workers AI (last-resort LLM).
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
# Primary TEXT model (script gen / JSON structuring). llama-3.3-70b-instruct
# is FREE on Workers AI, ~2x faster than gemma-4-26b, produces no empty-content
# retries (gemma's reasoning mode frequently ate the token budget and blanked),
# and its viral-shorts output is stronger (measured: 8.2s vs 16.1s+ on the same
# Arnold-style prompt).
CLOUDFLARE_MODEL = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")

# Groq — 3 keys: rotate on failure; Cloudflare is the only last resort.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY_SECOND = os.getenv("GROQ_API_KEY_SECOND")
GROQ_API_KEY_THIRD = os.getenv("GROQ_API_KEY_THIRD")
GROQ_API_KEY_BACKUP = os.getenv("GROQ_API_KEY_BACKUP")  # Legacy alias
# TEXT model (script/research): qwen is the fast on-demand service model, used
# EVERYWHERE by default. gpt-oss-120b is the automatic fallback, tried only
# when qwen is down/exhausted on all keys (the retry chain treats empty
# content as a failure, so a dead/unavailable model is never silently empty).
GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_MODEL_FALLBACK = "openai/gpt-oss-120b"

# Gemini removed from the chain (no Google in this project). The fallback after
# the 3 Groq keys is Cloudflare only.

CF_ENDPOINT = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"


class CfClientError(RuntimeError):
    """Non-retryable Cloudflare client error (401/403/400)."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"Cloudflare HTTP {status}: {detail}")

_primary = Groq(api_key=GROQ_API_KEY)
_rotating = 0
_clients = []
for _k in (GROQ_API_KEY, GROQ_API_KEY_SECOND, GROQ_API_KEY_THIRD, GROQ_API_KEY_BACKUP):
    if _k and _k not in {c.api_key for c in _clients}:
        _clients.append(Groq(api_key=_k))
_clients = _clients or [_primary]

_attempted = set()
_last_good = 0  # last Groq key that succeeded -> next call starts here


def _groq_keys_present() -> bool:
    """True when at least one Groq key is configured in .env.

    The central LLM seam (src.agents.common.llm) uses this to decide between
    Groq and local Ollama: Groq is only ever used when a real key exists.
    """
    return any(k for k in (
        GROQ_API_KEY, GROQ_API_KEY_SECOND, GROQ_API_KEY_THIRD, GROQ_API_KEY_BACKUP
    ) if k)


def _get_client():
    return _clients[_rotating % len(_clients)]


def _cf_is_vision(messages: list) -> bool:
    """True when any message carries an inline image payload."""
    return any(
        isinstance(c.get("content"), list)
        and any(p.get("type") == "image_url" for p in c["content"])
        for c in messages
        if isinstance(c, dict)
    )


def _cf_budget(attempt: int, max_tokens: int | None, is_vision: bool) -> int:
    """Per-attempt token budget. Reasoning model: reserve headroom for
    reasoning_content. Vision calls (big image prompts) need far more room
    after the reasoning pass, and empty-content retries double the budget.
    """
    floor = 2048 if is_vision else 1024  # floor for structured JSON output
    return max(max_tokens or 0, floor) * (2 ** attempt)


def _cf_attempt(url: str, headers: dict, kwargs: dict, attempt: int, tag: str) -> str | None:
    """One Cloudflare post. Returns content, or None when transient/busy/empty
    so the caller may retry; raises CfClientError (400-class) or RuntimeError
    (empty content at last attempt) immediately."""
    resp = requests.post(url, headers=headers, json=kwargs, timeout=180)
    if resp.status_code in (429, 503) or resp.status_code >= 500:
        wait = 2 ** attempt
        print(f"    [{tag}] Cloudflare {resp.status_code} (busy) - waiting {wait}s...")
        return None
    if resp.status_code >= 400:
        # Client-side errors (401/403/400) will not recover on retry: raise
        # immediately so callers fall back to Groq without wasted waits.
        raise CfClientError(resp.status_code, resp.text[:200])
    content = (resp.json()["choices"][0]["message"].get("content") or "").strip()
    if not content:
        wait = 2 ** attempt + 2
        print(f"    [{tag}] Cloudflare returned empty content (reasoning ate the budget?) - waiting {wait}s before retry...")
        return None
    return content


def call_cloudflare(messages: list, temperature: float = 0.7, model: str = None,
                    max_retries: int = 3, max_tokens: int = None, tag: str = "cf") -> str:
    """OpenAI-compatible Chat Completions against Cloudflare Workers AI.

    Last-resort provider (only reached when all Groq keys fail). The model is
    fast and busy under load, so requests retry with exponential backoff; raise
    on persistent failure so callers can fall back.
    """
    if model is None:
        model = CLOUDFLARE_MODEL
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise RuntimeError("[cf] CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID missing from .env")

    url = CF_ENDPOINT.format(account_id=CLOUDFLARE_ACCOUNT_ID)
    headers = {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}", "Content-Type": "application/json"}
    is_vision = _cf_is_vision(messages)

    for attempt in range(max_retries):
        kwargs = dict(
            model=model, messages=messages, temperature=temperature,
            max_completion_tokens=_cf_budget(attempt, max_tokens, is_vision),
        )
        try:
            content = _cf_attempt(url, headers, kwargs, attempt, tag)
            if content:
                return content
        except CfClientError:
            raise
        except Exception as e:
            print(f"    [{tag}] Attempt {attempt + 1}/{max_retries}: {e}")
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
        else:
            break

    raise RuntimeError(f"[{tag}] Cloudflare failed after {max_retries} retries")


def call_text(messages: list, temperature: float = 0.7, model: str = None,
              max_retries: int = 3, max_tokens: int = None, tag: str = "text") -> str:
    """Default text-generation seam: Groq first, Cloudflare last.

    Groq is the primary path (fast, 3 rotating keys). Cloudflare remains as a
    last resort — it only runs when all Groq keys are exhausted, so nothing
    about output is lost. No Gemini/Google anywhere in the chain.
    """
    groq_model = model or GROQ_MODEL
    try:
        return call_groq(messages, temperature=temperature, model=groq_model,
                         max_retries=max_retries, max_tokens=max_tokens, tag=tag)
    except Exception as e:
        print(f"    [{tag}] Groq unavailable ({str(e)[:120]}) - falling back to Cloudflare")
    cf_model = model or CLOUDFLARE_MODEL
    return call_cloudflare(messages, temperature=temperature, model=cf_model,
                           max_retries=max_retries, max_tokens=max_tokens, tag=tag)


def call_groq(messages: list, temperature: float = 0.7, model: str = None,
              max_retries: int = 3, max_tokens: int = None, tag: str = "groq") -> str:
    """Chat completions with 3-key rotation + model fallback.

    Primary model is GROQ_MODEL (qwen) by default; GROQ_MODEL_FALLBACK
    (gpt-oss-120b) is tried only when qwen fails on every key, so calls never
    die just because the primary service model is down. An explicit ``model``
    arg uses that model alone (no automatic fallback).

    Retries only on rate-limit/413/token errors (exponential backoff 1s, 2s,
    4s) up to ``max_retries`` on the same key; only then does it rotate to the
    next Groq key, and once all keys have been tried and failed on a model it
    moves to the fallback model. Non-retriable failures and empty responses
    rotate keys immediately. Returns the stripped text.
    """
    global _rotating, _last_good
    models = [model] if model else [GROQ_MODEL, GROQ_MODEL_FALLBACK]
    _model_err = None
    for model in models:
        _model_err = None
        for pos in range(len(_clients)):
            key_idx = (_last_good + pos) % len(_clients)
            _rotating = key_idx
            _attempted.add(key_idx)
            for attempt in range(max_retries):
                try:
                    kwargs = dict(model=model, messages=messages, temperature=temperature)
                    if max_tokens:
                        # The on-demand qwen model caps OUTPUT tokens at 1000/min, so a
                        # caller asking for e.g. max_tokens=8192 (script JSON) would be
                        # rejected at request time. Clamp to the limit ONLY for that
                        # model; standard models accept larger outputs, and callers
                        # downsample gracefully on partial JSON anyway.
                        if "qwen3.8-27b" in model:
                            kwargs["max_tokens"] = min(max_tokens, 1000)
                        else:
                            kwargs["max_tokens"] = min(max_tokens, 16384)
                    response = _get_client().chat.completions.create(**kwargs)
                    content = (response.choices[0].message.content or "").strip()
                    if content:
                        _last_good = key_idx
                        return content
                    print(f"    [{tag}] Key {key_idx + 1}/{len(_clients)} returned empty content")
                    _model_err = "empty content from groq"
                    break  # empty content on this key -> rotate to next key
                except Exception as e:
                    _model_err = str(e)[:120]
                    error_msg = str(e).lower()
                    retriable = ("rate_limit" in error_msg or "413" in error_msg
                                 or "tokens" in error_msg or "429" in error_msg)
                    print(f"    [{tag}] Key {key_idx + 1}/{len(_clients)} attempt {attempt + 1}/{max_retries} failed ({_model_err})")
                    if retriable and attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"    [{tag}] retrying same key in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    if retriable:
                        wait_time = 2 ** attempt
                        print(f"    [{tag}] all attempts failed - next key in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"    [{tag}] rotating to next key...")
                    break  # exhausted retries on this key -> rotate
        if _model_err and len(models) > 1:
            print(f"    [{tag}] {model} failed on all keys - trying fallback model")
            continue
        break
    raise RuntimeError(f"[{tag}] all {len(_clients)} Groq keys failed (last: {_model_err})")