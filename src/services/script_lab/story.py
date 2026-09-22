"""Stage 2 - STORY: facts + angle -> gripping short spoken script (plain prose).

No JSON here on purpose: the model narrates, and quality is enforced by
scoring a small best-of-N candidate pool (`_story_score`) so invented numbers
or wrong lengths are rejected before theme re-voicing.
"""
from __future__ import annotations

import re

from src.config.script_lab_prompts import GOLD_PATTERNS, STORY_PROMPT
from src.services.script_lab.facts import _facts_text, extract_angle
from src.services.script_lab.llm import _local
from src.services.script_lab.numbers import _canon_numbers, _fact_numbers
from src.services.script_lab.text import _count_sentences, _render


def _story_prompt(facts: list[dict], angle: dict) -> str:
    pre = f"{angle.get('reversal')}".strip()
    return _render(
        STORY_PROMPT,
        reversal_hint=pre if len(pre) > 8 else "what the public believed was not what existed",
        fooled_hint=f"{angle.get('fooled')}",
        hook_ticket_hint=f"{angle.get('hook_ticket')}",
        pairs_hint=" | ".join(f"{c}" for c in angle.get("contrast_pairs", [])),
        beats_hint=" -> ".join(f"{b}" for b in angle.get("beat_order", [])),
        GOLD_PATTERNS_BLOCK=GOLD_PATTERNS,
        facts=_facts_text(facts),  # no [kind=... value=...] annotations: the story
                                   # stage parroted those brackets verbatim
    )


def _story_score(prose: str, need: set[float]) -> tuple:
    """Quality score for a story candidate used for best-of-N selection.

    +10 per fact number that actually appears, -20 per invented number (a
    number that is not in the facts is a corruption risk, not flavor), plus a
    size term that prefers 8-12 sentences and 180-degrees prefers ~10.
    """
    have = set(_canon_numbers(prose))
    used = len(need & have)
    invented = len(have - need)
    n = _count_sentences(prose)
    size = -(abs(n - 10)) - (4 if not 8 <= n <= 12 else 0)
    return used * 10 - invented * 20 + size, n


def write_story(topic: str, facts: list[dict]) -> str:
    print("\n    ANGLE (frame-lock)")
    angle = extract_angle(facts)
    for k, v in angle.items():
        print(f"      {k}: {v}")
    base = _story_prompt(facts, angle)
    need = set(_fact_numbers(facts))

    banned = ["instead of reality", "instead of none", "instead of a single", " vs "]
    tighten = (
        "One sentence per line, 8 to 12 sentences total. Every number you use "
        "must be EXACTLY one of the facts' numbers - never invent or round them."
    )
    candidates: list[str] = []
    prose = _local(base, temperature=0.6, tag="story")
    candidates.append(prose)
    if any(b in prose for b in banned):
        print("    [story] echo of hint tokens detected - adding a clean re-roll")
        candidates.append(
            _local(
                base
                + "\n\nYou wrote the raw hint tokens ('vs', 'instead of reality', etc.) into the narration. "
                "Rewrite: every contrast must be narrated in your own words. " + tighten,
                temperature=0.6, tag="story",
            )
        )
    while len(candidates) < 3:
        best, n = max(_story_score(p, need) for p in candidates)
        if best >= 20 and 8 <= n <= 12:
            break
        print(f"    [story] best candidate scores {best} ({n} sentences) - one more attempt")
        candidates.append(_local(base + "\n\n" + tighten, temperature=0.6, tag="story"))
    # pick best-score candidate (ties broken by the size term inside _story_score)
    best, _ = max(_story_score(p, need) for p in candidates)
    prose = max(candidates, key=lambda p: _story_score(p, need))
    n = _count_sentences(prose)
    print(f"    [story] selected best-of-{len(candidates)} (score {best}, {n} sentences)")
    if n < 8:
        prose = _local(
            base
            + "\n\nYour previous narration had only a few sentences. The minimum is 8. "
            "Expand the SAME story with more of the facts - still one sentence per line, still exact numbers."
            + " " + tighten,
            temperature=0.6, tag="story",
        )
        n = _count_sentences(prose)
    if n > 12:  # last resort: keep hook + first 11 sentences
        parts = [p for p in re.split(r"(?<=[.!?])\s+", prose.strip()) if p.strip()]
        print(f"    [story] engine-at-{n} - trimming deterministically to 12")
        prose = " ".join(parts[:12])
    return prose