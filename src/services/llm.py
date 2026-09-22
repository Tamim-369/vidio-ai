"""Single seam for LLM calls (Groq + Gemini + Cloudflare + Ollama).

Owns the provider lifecycle and fallback chain. The primary pathways are:

    call_text()    -> Groq (3 rotating keys) -> Gemini -> Cloudflare (last resort).
    call_groq()    -> Groq (3 rotating keys), then Gemini fallback.
    call_ollama    -> Ollama -> Groq -> local chain for offline systems.

Groq keys rotate across GROQ_API_KEY / GROQ_API_KEY_SECOND / GROQ_API_KEY_THIRD
on retriable or hard failures, and repeated calls pick up where the last one
succeeded, so a dead key is never re-hit first; when all three are exhausted
``call_groq`` falls back to Gemini (``call_gemini``, rotating
GEMINI_API_KEY_ONE..FIVE). Cloudflare is demoted to a final safety net since it
rate-limits (429) constantly and was the main reason research/script generation
felt like forever. No service builds its own client or retry loop: use
``call_text`` / ``call_groq`` / ``call_ollama`` and forget the plumbing.

Note: there is NO vision seam anymore. Image verification was removed from the
pipeline — image selection is driven by the per-line query agent, so model
description/verification calls (and their Groq/Gemini/Ollama vision variants)
are gone.
"""
import time

import requests
from groq import Groq
import ollama

from src.config.settings import (
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_MODEL,
    GROQ_API_KEY,
    GROQ_API_KEY_BACKUP,
    GROQ_API_KEY_SECOND,
    GROQ_API_KEY_THIRD,
    GROQ_MODEL,
    GEMINI_KEYS,
    GEMINI_MODEL,
    OLLAMA_MODEL,
)

# Local Ollama models tried only if the configured (possibly cloud) model is down.
LOCAL_OLLAMA_FALLBACKS = ["qwen3:1.7b", "phi4-mini:latest"]

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


def is_using_backup() -> bool:
    """True once a rotated Groq key (beyond the primary) has been used."""
    return len(_attempted) > 1


def _get_client():
    return _clients[_rotating % len(_clients)]


def call_cloudflare(messages: list, temperature: float = 0.7, model: str = None,
                    max_retries: int = 3, max_tokens: int = None, tag: str = "cf") -> str:
    """OpenAI-compatible Chat Completions against Cloudflare Workers AI.

    Primary provider pipeline. The model is a reasoning model (@cf/google/
    gemma-4-26b-a4b-it) so its light reasoning (~150-200 tokens) is emitted in
    ``reasoning_content`` and the actual answer lands in ``content`` — callers
    only ever see the answer. Retries with exponential backoff on rate limits /
    5xx / capacity, raising on persistent failure so callers can fall back.
    """
    if model is None:
        model = CLOUDFLARE_MODEL
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise RuntimeError("[cf] CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID missing from .env")

    url = CF_ENDPOINT.format(account_id=CLOUDFLARE_ACCOUNT_ID)
    headers = {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}", "Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            is_vision = any(
                isinstance(c.get("content"), list)
                and any(p.get("type") == "image_url" for p in c["content"])
                for c in messages
                if isinstance(c, dict)
            )
            kwargs = dict(model=model, messages=messages, temperature=temperature)
            # Reasoning model: reserve headroom for reasoning_content. Vision
            # calls (big image prompts) need far more room after the reasoning pass,
            # and empty-content retries double the budget each attempt.
            floor = 2048 if is_vision else 1024  # Increased floor for structured JSON output
            budget = max(max_tokens or 0, floor) * (2 ** attempt)
            kwargs["max_completion_tokens"] = budget
            resp = requests.post(url, headers=headers, json=kwargs, timeout=180)
            if resp.status_code == 429 or resp.status_code == 503 or resp.status_code >= 500:
                wait = 2 ** attempt
                print(f"    [{tag}] Cloudflare {resp.status_code} (busy) - waiting {wait}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
            elif resp.status_code >= 400:
                # Client-side errors (401/403/400) will not recover on retry: raise
                # immediately so callers fall back to Groq without wasted waits.
                raise CfClientError(resp.status_code, resp.text[:200])
            data = resp.json()
            content = (data["choices"][0]["message"].get("content") or "").strip()
            if content:
                return content
            wait = 2 ** attempt + 2
            print(f"    [{tag}] Cloudflare returned empty content (reasoning ate the budget?) - waiting {wait}s before retry...")
            if attempt < max_retries - 1:
                time.sleep(wait)
                continue
            raise RuntimeError(f"[{tag}] empty content after {max_retries} attempts")
        except CfClientError:
            raise
        except Exception as e:
            print(f"    [{tag}] Attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    [{tag}] waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"[{tag}] Cloudflare failed after {max_retries} retries")


def call_text(messages: list, temperature: float = 0.7, model: str = None,
              max_retries: int = 3, max_tokens: int = None, tag: str = "text") -> str:
    """Default text-generation seam: Groq first, Gemini, Cloudflare last.

    Groq is now the primary path (fast, 3 rotating keys with a Gemini fallback),
    so script/research calls stop paying the Cloudflare 429 retry wall on every
    single request. Cloudflare remains as a last resort — it only runs when all
    Groq keys AND Gemini are exhausted, so nothing about output is lost.
    """
    groq_model = model or GROQ_MODEL
    try:
        return call_groq(messages, temperature=temperature, model=groq_model,
                         max_retries=max_retries, max_tokens=max_tokens, tag=tag)
    except Exception as e:
        print(f"    [{tag}] Groq/Gemini unavailable ({str(e)[:120]}) - falling back to Cloudflare")
    cf_model = model or CLOUDFLARE_MODEL
    return call_cloudflare(messages, temperature=temperature, model=cf_model,
                           max_retries=max_retries, max_tokens=max_tokens, tag=tag)


def call_groq(messages: list, temperature: float = 0.7, model: str = None,
              max_retries: int = 3, max_tokens: int = None, tag: str = "groq") -> str:
    """Chat completions with 3-key rotation + Gemini final fallback.

    Retries only on rate-limit/413/token errors (exponential backoff 1s, 2s,
    4s). On a retriable failure it first rotates to the next Groq key; once all
    keys have been tried and failed it falls back to Gemini (``call_gemini``).
    Non-retriable failures advance keys immediately. Returns the stripped text.
    """
    global _rotating, _last_good
    if model is None:
        model = GROQ_MODEL

    retriable = None
    for pos in range(len(_clients)):
        key_idx = (_last_good + pos) % len(_clients)
        _rotating = key_idx
        _attempted.add(key_idx)
        try:
            for attempt in range(max_retries):
                kwargs = dict(model=model, messages=messages, temperature=temperature)
                if max_tokens:
                    # The on-demand qwen model caps OUTPUT tokens at 1000/min, so a
                    # caller asking for e.g. max_tokens=8192 (script JSON) would be
                    # rejected at request time. Clamp to the limit ONLY for that
                    # model; standard models (llama-3.3-70b-versatile) accept
                    # larger outputs, and callers downsample gracefully on partial
                    # JSON anyway.
                    if "qwen3.8-27b" in model:
                        kwargs["max_tokens"] = min(max_tokens, 1000)
                    else:
                        kwargs["max_tokens"] = min(max_tokens, 16384)
                response = _get_client().chat.completions.create(**kwargs)
                content = (response.choices[0].message.content or "").strip()
                if content:
                    _last_good = key_idx
                    return content
                print(f"    [{tag}] Key {key_idx + 1}/{len(_clients)} returned empty content - next key")
                raise RuntimeError("empty content from groq")
        except Exception as e:
            error_msg = str(e).lower()
            print(f"    [{tag}] Key {key_idx + 1}/{len(_clients)} failed ({str(e)[:120]})")
            retriable = ("rate_limit" in error_msg or "413" in error_msg
                         or "tokens" in error_msg or "429" in error_msg)
            if retriable and pos < len(_clients) - 1:
                wait_time = 2 ** min(pos, 3)
                print(f"    [{tag}] rotating to next Groq key in {wait_time}s...")
                time.sleep(wait_time)
            elif not retriable and pos < len(_clients) - 1:
                print(f"    [{tag}] rotating to next Groq key...")
        # keep going to next key regardless
    print(f"    [{tag}] All {len(_clients)} Groq keys failed - falling back to Gemini")
    return call_gemini(messages, temperature=temperature, model=None,
                       max_tokens=max_tokens, tag=tag)


def call_gemini(messages: list, temperature: float = 0.7, model: str = None,
                max_tokens: int = None, tag: str = "gemini") -> str:
    """Gemini Flash completion via google-genai, rotating GEMINI_KEYS.

    Used as the final fallback after the 3 Groq keys. Handles both plain text
    and multimodal (inline image) messages. Raises a RuntimeError when
    every Gemini key fails so the caller's own fallback chain still runs.
    """
    global _rotating
    if model is None:
        model = GEMINI_MODEL
    if not GEMINI_KEYS:
        raise RuntimeError("[gemini] no GEMINI_API_KEY_* configured in .env")

    last_err = None
    for key in GEMINI_KEYS:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=key)
            contents = []
            for c in messages:
                if not isinstance(c, dict):
                    continue
                role = "model" if c.get("role") == "assistant" else "user"
                parts = _gemini_parts(c.get("content"))
                contents.append(types.Content(role=role, parts=parts))
            config = {"temperature": temperature}
            if max_tokens:
                config["max_output_tokens"] = min(max_tokens, 8192)
            resp = client.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(**config),
            )
            text = (resp.text or "").strip()
            if text:
                return text
            last_err = "empty response"
        except Exception as e:
            last_err = str(e)[:160]
            print(f"    [{tag}] Gemini key failed ({last_err}) - trying next")
            time.sleep(1)
    raise RuntimeError(f"[{tag}] all Gemini keys failed: {last_err}")


def _gemini_parts(content):
    """Convert an OpenAI-style message content to google-genai Parts."""
    from google.genai import types
    if isinstance(content, str):
        return [types.Part(text=content)]
    parts = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text" and p.get("text"):
            parts.append(types.Part(text=p["text"]))
        elif p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                mime, _, b64 = url[5:].partition(";base64,")
                if b64:
                    parts.append(types.Part(inline_data=types.Blob(
                        mime_type=mime or "image/jpeg", data=b64)))
    return parts


def call_ollama(messages: list, temperature: float = 0.3, model: str = None,
                max_tokens: int = None) -> str:
    """Configured Ollama model -> Groq -> local Ollama fallback chain.

    A dead/misconfigured cloud model never blocks: each candidate failing just
    logs and moves on, and Groq is tried before the slower local models because
    it produces clean structured JSON.
    """
    candidates = [model] if model else []
    if OLLAMA_MODEL:
        candidates.append(OLLAMA_MODEL)  # from .env (could be a :cloud model)

    for m in candidates:
        try:
            response = ollama.chat(
                model=m,
                messages=messages,
                options={"temperature": temperature},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"    [ollama] {m} failed ({str(e)[:120]}) - trying next")

    # Groq before local: it's fast and produces clean structured JSON.
    try:
        print("    [ollama] cloud model unavailable - falling back to Groq")
        return call_groq(messages, temperature=temperature, max_tokens=max_tokens)
    except Exception:
        pass

    # Last resort: any local model for offline operation.
    for m in LOCAL_OLLAMA_FALLBACKS:
        try:
            response = ollama.chat(
                model=m,
                messages=messages,
                options={"temperature": temperature},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"    [ollama] {m} failed ({str(e)[:120]})")
    raise RuntimeError("No Ollama model available (configured, cloud, or local)")