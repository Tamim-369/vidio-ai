"""Single seam for LLM calls (Groq + Ollama) shared across the pipeline.

Owns the Groq client lifecycle: primary key, rate-limit -> backup-key switch,
and exponential backoff. Also owns the Ollama -> Groq -> local-Ollama fallback
chain used by script_builder and callers (`_call_ollama`).

The point of this module: no service ever builds its own Groq client or retry
loop again. Use ``call_groq`` / ``call_ollama`` and forget the plumbing.
"""
import time

from groq import Groq
import ollama

from src.config.settings import (
    GROQ_API_KEY,
    GROQ_API_KEY_BACKUP,
    GROQ_MODEL,
    OLLAMA_MODEL,
)

# Local Ollama models tried only if the configured (possibly cloud) model is down.
LOCAL_OLLAMA_FALLBACKS = ["qwen3:1.7b", "phi4-mini:latest"]

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
                kwargs["max_tokens"] = max_tokens
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