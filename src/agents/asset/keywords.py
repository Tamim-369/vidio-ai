"""Local topic-keyword generation (was Groq, tag "kw").

Turns a video topic into ~6 findable search phrases with the local model and
falls back to a stopword heuristic when the model is down — so keyword
generation never touches a cloud provider and never blocks asset fetching.
"""
from __future__ import annotations

from src.agents.asset.local import complete, extract_json_block, strip_thinking
from src.agents.asset.prompts import KEYWORD_PROMPT, N_KEYWORDS

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with",
    "from", "by", "that", "this", "how", "what", "why", "was", "were", "had",
    "his", "her", "their", "they", "it's", "its", "about", "during", "after",
    "before", "over", "under", "into", "against", "story", "video", "true",
    "real", "shocking", "secret", "secrets", "history", "mystery", "mysteries",
    "always", "common", "look", "aren't", "isn't", "don't", "not", "top", "5",
    "4", "3", "2", "1", "dark", "truth", "human",
}


def _heuristic(topic: str) -> list:
    words = [w for w in topic.lower().split() if w.strip(".,?!'\"") not in _STOPWORDS]
    core = " ".join(words[:4])
    return [core] if core else [topic]


def _as_keyword_list(raw: str | None) -> list | None:
    """Accept a JSON list, a {"keywords": [...]} dict, or plain line list."""
    if not raw:
        return None
    for candidate in (extract_json_block(raw), raw):
        if isinstance(candidate, dict):
            for key in ("keywords", "phrases", "query"):
                val = candidate.get(key)
                if isinstance(val, list):
                    return val
            for val in candidate.values():
                if isinstance(val, list):
                    return val
        elif isinstance(candidate, str):
            return _kv_lines(candidate)
    return None


def topic_keywords(topic: str, n: int = N_KEYWORDS) -> list:
    try:
        raw = strip_thinking(complete(KEYWORD_PROMPT.format(topic=topic), tag="kw", temperature=0.6))
        kws = _as_keyword_list(raw)
        if kws:
            cleaned = []
            for k in kws:
                if isinstance(k, str):
                    k = k.strip().strip('"').strip("-").strip(".").strip()
                    if 2 <= len(k.split()) <= 8:
                        cleaned.append(k)
            if len(cleaned) >= 4:
                return cleaned[: n]
    except Exception:
        pass
    return _heuristic(topic)


def _kv_lines(raw: str) -> list | None:
    lines = [ln.strip().strip('"').strip("-").strip(".") for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln and len(ln.split()) <= 8]
    return lines if lines else None