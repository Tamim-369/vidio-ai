"""Generates deadpan fake-wisdom jokes for the whole channel: a video is two or
three one- or two-sentence jokes on one subject, not a documentary about a topic.

- The prompt is the user's specification and is not ours to reword. The subject
  comes from the character who is narrating (war for Don Tzu, weight lifting for
  Brolexander, money for Andru Tatte), so the joke and the voice always agree.
- Groq only: call_groq(allow_fallback=False) so a dead key raises rather than
  silently yielding Gemini output. GROQ_QUOTE_MODEL is kept separate from the
  script/research GROQ_MODEL.
- Batch, then filter: one reply yields several candidates so a rejection costs
  no extra round trip.
- No quote history: nothing is persisted between runs. The prompt is asked for
  fresh material every time and each call is independent. The only duplication
  check is within a single call, because two identical quotes in one video
  would render as two identical cards.
- Rejection is local: the prompt is never edited and no extra instructions are
  appended, so a bad candidate is filtered here and the round is simply
  resampled."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from src.agents.completion import llm
from src.agents.quotes.json_parse import _loads_json
from src.agents.quotes.prompt import get_prompt

# Quotes run on their own model so they can change independently of the
# script/research model. Groq-only by policy: generate_quotes passes
# allow_fallback=False so a dead key raises instead of silently returning
# another provider's output.
GROQ_QUOTE_MODEL = os.getenv("GROQ_QUOTE_MODEL", "openai/gpt-oss-20b")

# How many candidates to request per round. More than we need so rotation works
# and rejects are cheap.
BATCH = 6

# Per-candidate output budget, plus a fixed allowance for the model's reasoning.
# gpt-oss emits reasoning tokens before the answer, so a quote costs well more
# than its own text; a budget that only fits the answers truncates the response
# and comes back empty.
_MAX_TOKENS_PER_QUOTE = 160
_MAX_TOKENS_OVERHEAD = 1024
MAX_ROUNDS = 3

# The prompt asks for "preferably 10-35 words", which is roughly 55-200
# characters. The cap has to clear that window or valid jokes are thrown away:
# the first live run of the new prompt produced 131-170 character quotes, all
# of which an 80-char cap would have rejected. The card shrinks its type from
# 64px down to 30px to fit, so a 200-character quote still renders as readable
# type in the top 40% of the frame.
# The floor exists because terse one-liners leave a lot of dead frame and read
# as a caption rather than a joke.
# Fallback window, used only when a caller rejects against a bare pool. The
# window actually applied comes from the character's own prompt module, because
# each prompt states its own limit.
MAX_QUOTE_CHARS = 200
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

# The prompt's own examples of a failed joke are real wisdom with the twist
# removed ("Peace is simply the absence of war"), and a live run produced
# exactly that. Those are mechanically recognisable even though they are not
# jokes, so they are rejected here rather than narrated.
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
    """One accepted joke."""

    text: str


# --- parsing / filtering -----------------------------------------------------

# Models emit Unicode lookalikes for characters that the rest of the pipeline
# cannot handle: TTS drops or mispronounces a non-breaking hyphen ("long-term"
# comes out as one word) and the card font may not carry the glyph at all.
_CONFUSABLE_PUNCTUATION = {
    # Hyphen-minus lookalikes, including the en dash models reach for when they
    # mean a hyphen. The em dash is left alone; that one is real punctuation.
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "⁃": "-", "−": "-",
    # Spaces that are not spaces.
    " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",
    # Invisible characters that would otherwise pad the character count.
    "": "", "‌": "", "‍": "", "﻿": "",
}
_CONFUSABLE_RE = re.compile("|".join(map(re.escape, _CONFUSABLE_PUNCTUATION)))


def _strip_wrapper(text: str) -> str:
    """Remove numbering, bullets and surrounding quotes the model may add."""
    t = _CONFUSABLE_RE.sub(lambda m: _CONFUSABLE_PUNCTUATION[m.group()], text or "")
    t = t.strip()
    t = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", t)
    t = re.sub(r'^["“‘\']|["”’\']$', "", t).strip()
    # Collapse a doubled wrapper, e.g. `"1. "Quote text""`.
    t = re.sub(r'^\d+[.)]\s*["“‘\']', "", t).strip()
    return t


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
                out.append({"quote": _strip_wrapper(str(text))})
            elif isinstance(item, str):
                out.append({"quote": _strip_wrapper(item)})
    elif isinstance(parsed, dict) and parsed.get("quote"):
        out.append({"quote": _strip_wrapper(str(parsed["quote"]))})
    else:
        # Plain text fallback: one quote per non-empty line.
        for line in raw.splitlines():
            line = line.strip()
            if not line or _META.search(line):
                continue
            out.append({"quote": _strip_wrapper(line)})

    return [c for c in out if c["quote"]]


def reject_reason(cand: dict, seen: set, min_chars: int = MIN_QUOTE_CHARS,
                  max_chars: int = MAX_QUOTE_CHARS) -> str:
    """Return why a candidate is unusable, or "" to accept it.

    These are the mechanically checkable rules: no meta-commentary, not a repeat
    of another quote in this same video, and a length inside the window the
    character's prompt asked for.

    ``seen`` holds the normalised text of everything already accepted for the
    current call. It is deliberately not persisted: a duplicate is only a
    visible defect when both copies land on the same video.
    """
    text = (cand.get("quote") or "").strip()

    if not text:
        return "empty"
    if _META.search(text):
        return "meta-commentary"
    if len(text) < min_chars:
        return "too short"
    if len(text) > max_chars:
        return "too long"
    if "\n" in text:
        return "multi-line (must be one beat)"

    if _FACT_YEAR.search(text) or _FACT_OPENING.search(text) \
            or _FACT_ASSERTION.search(text) or _FACT_NARRATIVE.search(text):
        return "reads like a fact, not a joke"

    if _normalise(text) in seen:
        return "duplicate within this video"

    return ""


def _normalise(text: str) -> str:
    """Loose key for duplicate detection: case, spacing and trailing stop."""
    t = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


# --- generation --------------------------------------------------------------

def _build_messages(n: int, subject: str, prompt_mod) -> list:
    """Return the user's prompt, filled in, as the single user message.

    The prompt is sent as-is. It already states the length hint ("preferably
    10-35 words"), the output format and the self-check, so restating any of
    that here would only be a second, drifting copy of the specification.
    """
    user = prompt_mod.build_prompt(subject, n)
    return [{"role": "user", "content": user}]


def generate_quotes(n: int = 1, subject: str = "",
                    explain: bool = False, character: str = "") -> list:
    """Return up to ``n`` fresh, validated jokes on ``subject``.

    explain=True prints every candidate the model produced that was *not*
    accepted, with the reason. Without it a candidate silently vanishes and a
    short result is indistinguishable from a stingy model, which is the usual
    reason the prompt looks like it is "not working".

    Raises RuntimeError if no round produced enough acceptable quotes, rather
    than returning a weak one: the prompt treats real wisdom as a failure, so
    failing loudly beats narrating something that actually makes sense.
    """
    prompt_mod = get_prompt(character)
    min_chars, max_chars = prompt_mod.MIN_CHARS, prompt_mod.MAX_CHARS

    if character and prompt_mod.__name__.endswith("prompt_shared"):
        # Silently borrowing the shared prompt would hide that this character
        # has no prompt of its own yet.
        print(f"    [quotes] {character} has no prompt yet, using the shared one")

    accepted: list = []
    seen: set = set()

    for round_no in range(1, MAX_ROUNDS + 1):
        need = n - len(accepted)
        if need <= 0:
            break
        want = max(BATCH, need * 2)
        messages = _build_messages(want, subject, prompt_mod)
        # max_tokens has to scale with the request. A fixed budget looks fine
        # for a 1-2 quote call and silently breaks a 6 quote one: gpt-oss spends
        # tokens reasoning first, so asking for 12 candidates inside 2048 tokens
        # runs out mid-answer, returns finish_reason="length" with an empty
        # message, and every key then fails identically. Reserve room for the
        # answers plus the model's reasoning overhead.
        max_tokens = _MAX_TOKENS_OVERHEAD + want * _MAX_TOKENS_PER_QUOTE
        raw = llm.call_groq(
            messages,
            temperature=0.95,          # variety matters more than consistency
            model=GROQ_QUOTE_MODEL,
            max_tokens=max_tokens,
            tag=f"quotes r{round_no}",
            allow_fallback=False,      # Groq only, never Gemini
        )

        for cand in parse_candidates(raw):
            if len(accepted) >= n:
                break
            reason = reject_reason(cand, seen, min_chars, max_chars)
            if reason:
                if explain:
                    print(f"   ✗ dropped ({reason}): {cand['quote']}")
                continue
            quote = Quote(text=cand["quote"])
            accepted.append(quote)
            seen.add(_normalise(quote.text))

    if len(accepted) < n:
        raise RuntimeError(
            f"quote agent produced {len(accepted)}/{n} usable quotes after "
            f"{MAX_ROUNDS} rounds"
        )

    return accepted
