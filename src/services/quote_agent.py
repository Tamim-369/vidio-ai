"""Quote agent — Groq-only generation of short funny "anti-wisdom" quotes.

This is the content source for the whole channel. It replaces the retired
topic -> research -> script pipeline: instead of a documentary about a
historical topic, a video is one to two one-beat jokes.

Design notes
------------
* **Groq only.** Calls ``llm.call_groq(..., allow_fallback=False)`` so a dead
  key raises rather than silently yielding Gemini output, which would violate
  the "quotes come from Groq" requirement. The model is
  ``settings.GROQ_QUOTE_MODEL`` (``GROQ_QUOTE_MODEL`` env, default
  ``qwen3.8-27b``), kept separate from the script/research ``GROQ_MODEL``.

* **Batch, then filter.** One reply yields several candidates so formats can
  rotate within a batch as the prompt requires, and so a rejected candidate
  does not cost an extra round trip. The model self-checks per its own prompt;
  the filters here catch what the model gets wrong anyway.

* **Accumulating pool.** Accepted quotes and the figures they twist are written
  to a JSON state file so later videos avoid repeats and rotate widely. The
  prompt's "rotate widely / don't reuse" rules are only enforceable across
  videos if that history is persisted.

* **Rejection is silent to the narrator.** The prompt says a bad quote must be
  discarded and regenerated rather than narrated, so a rejected candidate is
  never returned; it only triggers another round with feedback.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict

from src.config.quote_prompt import (
    OUTPUT_FORMAT,
    QUOTE_PROMPT,
    SOURCE_FIGURES,
)
from src.config.settings import GROQ_QUOTE_MODEL
from src.services import llm
from src.utils.text_helpers import _loads_json

# Durable, inside src/ so the pool is not wiped by file_helpers.cleanup_temp()
# and does not add an untracked file at the repo root. Override with
# QUOTE_STATE_FILE.
STATE_FILE = os.getenv("QUOTE_STATE_FILE", "src/state/used_quotes.json")

# How many candidates to request per round. More than we need so rotation works
# and rejects are cheap.
BATCH = 6
MAX_ROUNDS = 3
# The card sets the quote in the top 40% of the frame, so anything long wraps
# into a wall of small type. 80 chars is roughly a 3-4 line card at the sizes
# the renderer uses. Over-long candidates are rejected here and regenerated
# rather than rendered, so the cap holds at the source instead of being
# truncated mid-sentence on the card.
# The floor exists because terse one-liners leave a lot of dead frame and read as
# a caption rather than a punchline, so the sweet spot is a real sentence.
MAX_QUOTE_CHARS = 80
MIN_QUOTE_CHARS = 30

# Meta-commentary the prompt forbids. A candidate carrying any of these is
# dropped rather than stripped: the model self-correcting mid-output means the
# rest of the reply is untrustworthy too.
_META = re.compile(
    r"(self[- ]?check|as an ai|i (?:have )?(?:re)?generated|"
    r"discard(?:ing|ed)?|regenerat(?:e|ing)|format label|no format|"
    r"note:|explanation:|rewrite:|revised:)",
    re.IGNORECASE,
)

_VALID_FORMATS = {"twisted_proverb", "fake_attribution", "one_liner", "crude"}

# A live run showed the model still slipping real historical statements past the
# prompt's self-check ("The Battle of Thermopylae was a tactical victory, not a
# strategic one."). Those are mechanically recognisable even though they are
# not jokes, so they are rejected here rather than narrated.
_FACT_YEAR = re.compile(r"\b(?:\d{1,3}\s*(?:BC|AD)\b|[12]\d{3}\b)", re.IGNORECASE)
_FACT_OPENING = re.compile(
    r"^(?:the|a|an)\s+[A-Z][\w'-]*\s+"
    r"(?:battle|war|siege|empire|king|queen|reign|period|era|revolution|"
    r"treaty|alliance|dynasty|campaign|expedition|conquest)\b"
)
_FACT_ASSERTION = re.compile(
    r"\b(?:was|is|were|are)\s+(?:a|an|the)\b[^.?!]{0,60}?\bnot\s+(?:a|an|the)\b",
    re.IGNORECASE,
)
_FACT_NARRATIVE = re.compile(
    r"\b(?:is|was)\s+(?:remembered|regarded|considered|known)\s+(?:as|for)\b",
    re.IGNORECASE,
)


@dataclass
class Quote:
    """One accepted quote plus the bookkeeping used for future rotation."""

    text: str
    format: str = "one_liner"
    source: str = ""
    figure: str = ""
    video: str = ""


@dataclass
class _Pool:
    quotes: list = field(default_factory=list)
    figures: list = field(default_factory=list)


# --- state -------------------------------------------------------------------

def _load_pool(path: str = None) -> _Pool:
    path = path or STATE_FILE
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return _Pool()
    if not isinstance(data, dict):
        return _Pool()
    return _Pool(
        quotes=[q for q in data.get("quotes", []) if isinstance(q, dict)],
        figures=[f for f in data.get("figures", []) if isinstance(f, str)],
    )


def _save_pool(pool: _Pool, path: str = None) -> None:
    path = path or STATE_FILE
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"quotes": pool.quotes, "figures": pool.figures}, f, indent=2)


def _remember(pool: _Pool, quote: Quote) -> None:
    pool.quotes.append(asdict(quote))
    if quote.figure and quote.figure not in pool.figures:
        pool.figures.append(quote.figure)


# --- parsing / filtering -----------------------------------------------------

def _strip_wrapper(text: str) -> str:
    """Remove numbering, bullets and surrounding quotes the model may add."""
    t = (text or "").strip()
    t = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", t)
    t = re.sub(r'^["“‘\']|["”’\']$', "", t).strip()
    # Collapse a doubled wrapper, e.g. `"1. "Quote text""`.
    t = re.sub(r'^\d+[.)]\s*["“‘\']', "", t).strip()
    return t


def _detect_figure(quote: str, source: str = "") -> str:
    """Find which historical figure a quote twists, for rotation tracking.

    Checks the declared ``source`` first, then the quote body for a
    "Name said ..." style attribution.
    """
    hay = f"{source} {quote}".lower()
    for fig in SOURCE_FIGURES:
        if fig in hay:
            return fig
    return ""


def parse_candidates(raw: str) -> list:
    """Extract candidate dicts from a model reply.

    Tolerates the requested JSON array, a bare JSON array, fenced JSON, and
    plain one-quote-per-line text. Returns [] when nothing usable is present.
    """
    if not raw or not raw.strip():
        return []
    parsed = None
    try:
        parsed = _loads_json(raw)
    except ValueError:
        parsed = None

    out = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                text = item.get("quote") or item.get("text") or ""
                out.append({
                    "quote": _strip_wrapper(str(text)),
                    "format": str(item.get("format") or "").strip().lower(),
                    "source": str(item.get("source") or "").strip(),
                })
            elif isinstance(item, str):
                out.append({"quote": _strip_wrapper(item),
                            "format": "", "source": ""})
    elif isinstance(parsed, dict) and parsed.get("quote"):
        out.append({"quote": _strip_wrapper(str(parsed["quote"])),
                    "format": str(parsed.get("format") or "").strip().lower(),
                    "source": str(parsed.get("source") or "").strip()})
    else:
        # Plain text fallback: one quote per non-empty line.
        for line in raw.splitlines():
            line = line.strip()
            if not line or _META.search(line):
                continue
            out.append({"quote": _strip_wrapper(line), "format": "", "source": ""})

    return [c for c in out if c["quote"]]


def reject_reason(cand: dict, pool: _Pool) -> str:
    """Return why a candidate is unusable, or "" to accept it.

    These are the mechanically checkable rules from the prompt: no
    meta-commentary, no duplicate of something already used, sane length, and
    a known format tag.
    """
    text = (cand.get("quote") or "").strip()

    if not text:
        return "empty"
    if _META.search(text):
        return "meta-commentary"
    if len(text) < MIN_QUOTE_CHARS:
        return "too short"
    if len(text) > MAX_QUOTE_CHARS:
        return "too long"
    if "\n" in text:
        return "multi-line (must be one beat)"

    if _FACT_YEAR.search(text) or _FACT_OPENING.search(text) \
            or _FACT_ASSERTION.search(text) or _FACT_NARRATIVE.search(text):
        return "reads like a fact, not a joke"

    norm = _normalise(text)
    for q in pool.quotes:
        if _normalise(q.get("text", "")) == norm:
            return "already used"

    fmt = cand.get("format") or ""
    if fmt and fmt not in _VALID_FORMATS:
        return f"unknown format '{fmt}'"

    return ""


def _normalise(text: str) -> str:
    """Loose key for duplicate detection: case, spacing and trailing stop."""
    t = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _figure_is_stale(figure: str, pool: _Pool) -> bool:
    """True when this figure was used more recently than every alternative.

    Keeps a single twisted-proverb video from hammering Sun Tzu every time
    while still allowing reuse once the pool has wrapped.
    """
    if not figure or not pool.figures:
        return False
    remaining = [f for f in SOURCE_FIGURES if f not in pool.figures]
    if remaining:
        return False  # unused figures exist -> prefer them over reuse
    return pool.figures[-1] == figure


# --- generation --------------------------------------------------------------

def _build_messages(rejected: list, pool: _Pool, n: int) -> list:
    user = QUOTE_PROMPT + OUTPUT_FORMAT.format(n=n)
    if pool.quotes:
        recent = [q.get("text", "") for q in pool.quotes[-40:]]
        user += ("\n\nAlready used — do NOT repeat or paraphrase any of these:\n"
                 + "\n".join(f"- {t}" for t in recent))
    if pool.figures:
        user += ("\n\nFigures already twisted (prefer the others): "
                 + ", ".join(pool.figures[-12:]))
    if rejected:
        user += ("\n\nThese were rejected — do not repeat them, and do not include "
                 "any commentary about them:\n"
                 + "\n".join(f"- {r}" for r in rejected))
    # The prompt's own length hint, restated as hard numbers. Without this the
    # model keeps writing 150-character jokes that are then thrown away, which
    # burns rounds and can starve a batch. The floor is restated too: a terse
    # 12-character line is just as wasteful as an over-long one.
    user += (f"\n\nHARD LIMIT: every quote must be between {MIN_QUOTE_CHARS} "
             f"and {MAX_QUOTE_CHARS} characters INCLUDING spaces and "
             f"punctuation. Count before you answer. Anything outside that "
             f"range is rejected and thrown away.")
    return [{"role": "user", "content": user}]


def generate_quotes(n: int = 1, pool_path: str = None) -> list:
    """Return up to ``n`` fresh, validated quotes and record them in the pool.

    Raises RuntimeError if no round produced enough acceptable quotes, rather
    than returning a weak quote: the prompt treats a bad quote as unusable, so
    failing loudly is better than narrating real wisdom.
    """
    pool = _load_pool(pool_path)
    rejected: list = []
    accepted: list = []

    for round_no in range(1, MAX_ROUNDS + 1):
        need = n - len(accepted)
        if need <= 0:
            break
        messages = _build_messages(rejected, pool, max(BATCH, need * 2))
        raw = llm.call_groq(
            messages,
            temperature=0.95,          # variety matters more than consistency
            model=GROQ_QUOTE_MODEL,
            max_tokens=2048,
            tag=f"quotes r{round_no}",
            allow_fallback=False,      # Groq only, never Gemini
        )

        for cand in parse_candidates(raw):
            if len(accepted) >= n:
                break
            reason = reject_reason(cand, pool)
            if reason:
                if cand.get("quote"):
                    rejected.append(cand["quote"])
                continue
            figure = _detect_figure(cand["quote"], cand.get("source", ""))
            if _figure_is_stale(figure, pool):
                continue
            quote = Quote(
                text=cand["quote"],
                format=cand.get("format") or "one_liner",
                source=cand.get("source", ""),
                figure=figure,
            )
            accepted.append(quote)
            _remember(pool, quote)

    if len(accepted) < n:
        raise RuntimeError(
            f"quote agent produced {len(accepted)}/{n} usable quotes after "
            f"{MAX_ROUNDS} rounds"
        )

    _save_pool(pool, pool_path)
    return accepted
