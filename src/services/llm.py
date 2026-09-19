"""Single seam for LLM calls (Cloudflare + Groq + Ollama) shared across the pipeline.

Owns the provider lifecycle and fallback chain. The primary pathway is:

    call_text()   -> Cloudflare Workers AI (gemma-4-26b) first, Groq second.

Cloudflare is the "workhorse" for script generation and structured JSON when
configured (token + account id in .env); Groq is the fast secondary. `call_groq`
remains for direct Groq-only calls (e.g. image verification fallback) and
`call_ollama` keeps the Ollama -> Groq -> local chain for offline systems.

The point of this module: no service ever builds its own client or retry loop
again. Use ``call_text`` / ``call_groq`` / ``call_ollama`` and forget the
plumbing.
"""
import time

import requests
from groq import Groq
import ollama

from src.config.settings import (
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_MODEL,
    CLOUDFLARE_VISION_MODEL,
    GROQ_API_KEY,
    GROQ_API_KEY_BACKUP,
    GROQ_MODEL,
    GROQ_VISION_MODEL,
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
_backup = None
_using_backup = False


def is_using_backup() -> bool:
    """True once a rate limit forced us onto the backup API key."""
    return _using_backup


def _get_client():
    global _backup, _using_backup
    if _using_backup and _backup:
        return _backup
    return _primary


def _switch_to_backup(tag: str) -> bool:
    """Switch the active client to the backup API key, if one is configured."""
    global _backup, _using_backup
    if GROQ_API_KEY_BACKUP and not _using_backup:
        print(f"    [{tag}] Switching to backup API key...")
        _backup = Groq(api_key=GROQ_API_KEY_BACKUP)
        _using_backup = True
        return True
    return False


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
            floor = 2048 if is_vision else 256
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
              max_retries: int = 3, max_tokens: int = None, tag: str = "llm",
              vision: bool = False) -> str:
    """Default text-generation seam: Cloudflare primary, Groq secondary.

    Text calls use ``CLOUDFLARE_MODEL`` (gemma-4-26b, high script quality);
    vision calls (image verify/refine) use ``CLOUDFLARE_VISION_MODEL``
    (llama-4-scout, ~5x faster on vision) which is multimodal.
    """
    cf_model = CLOUDFLARE_VISION_MODEL if vision else model or CLOUDFLARE_MODEL
    try:
        return call_cloudflare(messages, temperature=temperature, model=cf_model,
                               max_retries=max_retries, max_tokens=max_tokens, tag=tag)
    except Exception as e:
        print(f"    [{tag}] Cloudflare unavailable ({str(e)[:120]}) - falling back to Groq")
    groq_model = GROQ_VISION_MODEL if vision else model or GROQ_MODEL
    return call_groq(messages, temperature=temperature, model=groq_model,
                     max_retries=max_retries, max_tokens=max_tokens, tag=tag)


def call_groq(messages: list, temperature: float = 0.7, model: str = None,
              max_retries: int = 3, max_tokens: int = None, tag: str = "groq") -> str:
    """Chat completions with backup-key fallback + exponential backoff.

    Retries only on rate-limit/413/token errors (exponential backoff 1s, 2s, 4s),
    switching to the backup key on the first hit. Other failures re-raise
    immediately so callers see real errors. Returns the stripped completion text.
    """
    if model is None:
        model = GROQ_MODEL

    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, messages=messages, temperature=temperature)
            if max_tokens:
                # Groq's free tier (qwen/qwen3.8-27b) caps OUTPUT tokens at 1000
                # per minute for on_demand service. A caller asking for e.g.
                # max_tokens=8192 (script JSON conversion) is rejected at request
                # time — the fallback would ALWAYS throw. Clamp to the limit so a
                # Groq fallback can actually produce output. The structured-JSON
                # callers downsample gracefully (the parser recovers partial
                # fields), so this never silently corrupts.
                kwargs["max_tokens"] = min(max_tokens, 1000)
            response = _get_client().chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e).lower()
            print(f"    [{tag}] Attempt {attempt + 1}/{max_retries}: {e}")

            if "rate_limit" in error_msg or "413" in error_msg or "tokens" in error_msg:
                if attempt == 0 and not _using_backup and _switch_to_backup(tag):
                    print(f"    [{tag}] Retrying with backup key...")
                    continue  # retry immediately with backup
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"    [{tag}] Rate limit hit, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
            raise

    raise RuntimeError(f"[{tag}] Failed after {max_retries} retries")


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