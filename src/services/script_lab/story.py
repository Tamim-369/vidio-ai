"""Stage 2 - STORY: facts + angle -> gripping short spoken script (plain prose).

No JSON here on purpose: the model narrates, and quality is enforced by
scoring a small best-of-N candidate pool (`_story_score`) so invented numbers
or wrong lengths are rejected before theme re-voicing.
"""
from __future__ import annotations

import re

from src.config.script_lab_prompts import GOLD_PATTERNS, STORY_BEAT_PROMPT, STORY_PROMPT
from src.services.script_lab.facts import _facts_text, extract_angle
from src.services.script_lab.llm import _local
from src.services.script_lab.numbers import (
    _SPELLED_RE,
    _canon_numbers,
    _eval_spelled,
    _fact_numbers,
)
from src.services.script_lab.text import _count_sentences, _render

# Verbatim phrases from the GOLD_PATTERNS / ANGLE exemplars. A small local model
# happily pastes them wholesale into the narration ("the safest ship ever
# built...") instead of imitating their FORM. Any hit is a hard reject: the
# story is supposed to be ABOUT the facts, not a rewrite of the examples.
_VAGUE_QTY_RE = re.compile(
    r"\b(?:hundreds|thousands|millions|countless|lots|scores of|dozens|numerous|"
    r"many|a handful)\b",
    re.IGNORECASE,
)

_EXAMPLE_LEAKS = [
    "army that didn't exist", "army that was not there", "army that never existed",
    "safest ship", "maiden voyage", "inflatable tank", "1,100", "1,100 men",
    "tiny crew", "500 yards", "smoke and cloth", "what else is out there",
    "not being told", "an image of an army", "didn't exist. it fooled",
    "they were real", "in the dark. from",
]


def _example_leak(prose: str) -> int:
    low = (prose or "").lower()
    return sum(1 for flag in _EXAMPLE_LEAKS if flag.lower() in low)


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
    size term that prefers 8-12 sentences and 180-degrees prefers ~10. A
    REPETITION penalty (-4 per extra mention of a fact number beyond the first)
    stops the model from jamming the same figure into every sentence - the
    "over two thousand..." echo that turns a story into a broken record.
    """
    have = set(_canon_numbers(prose))
    used = len(need & have)
    invented = len(have - need)
    n = _count_sentences(prose)
    size = -(abs(n - 10)) - (4 if not 8 <= n <= 12 else 0)

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", prose.strip()) if s.strip()]
    counts: dict = {}
    for s in sentences:
        for num in _canon_numbers(s):
            counts[num] = counts.get(num, 0) + 1
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    leaks = _example_leak(prose)
    vague = len(_VAGUE_QTY_RE.findall(prose))
    return used * 10 - invented * 40 + size - 4 * repeats - 60 * leaks - 5 * vague, n


def _sanitize_angle(angle: dict) -> dict:
    """Drop any angle hint that came back as a pasted example phrase."""
    out = {}
    for k, v in angle.items():
        if _example_leak(str(v)):
            print(f"    [angle] {k} leaked an example phrase - blanked")
            out[k] = ""
        else:
            out[k] = v
    return out


_DEFAULT_BEATS = [
    "open on the contrast",
    "show how the accepted story held",
    "escalate with the scale of what really happened",
    "climax on the reveal",
]

_DEFAULT_ANGLE = {
    "reversal": "what the public believed was not what existed",
    "fooled": "the accepted story hid the real one",
}


def _patch_angle(angle: dict) -> dict:
    """Replace degenerate angle fields ('...', empty, ellipsis placeholders)
    with the generic defaults so a bad planning reply cannot hollow out every
    story it feeds."""
    out = dict(angle)
    for k, v in list(out.items()):
        if isinstance(v, str) and (not v.strip() or "..." in v or v.strip().lower() == "none"):
            print(f"    [angle] {k} came back empty/placeholder - using default")
            out[k] = _DEFAULT_ANGLE.get(k, "")
    if not out.get("hook_ticket"):
        out["hook_ticket"] = "a specific fact from the story that is true and shocking"
    beats = out.get("beat_order")
    if not isinstance(beats, list) or not any(
        isinstance(b, str) and b.strip() and "..." not in b for b in beats
    ):
        print("    [angle] beat_order empty - using default beats")
        out["beat_order"] = _DEFAULT_BEATS
    else:
        out["beat_order"] = [b for b in beats if isinstance(b, str) and b.strip() and "..." not in b]
    return out


def _digits_for_facts(prose: str, need: set[float]) -> str:
    """Deterministically rewrite spelled-out FACT figures into digit form.

    The small model keeps writing "seventy" / "two thousand" despite the AS
    DIGITS instruction. When a spelled run parses to a factual figure, it is
    replaced with its digit form so narration, captions and the fidelity audit
    all see the exact number. Spelled runs that parse to junk (a wrong year
    like "twenty-five") are left untouched.
    """
    def _fmt(v: float) -> str:
        return f"{v:,.0f}" if v >= 1000 else f"{v:g}"

    chunks, pos = [], 0
    for m in _SPELLED_RE.finditer(prose):
        toks = re.split(r"[\s-]+", m.group(0).strip().lower())
        if not toks:
            continue
        val = _eval_spelled(toks)
        if round(val, 3) not in need:
            continue
        chunks.append(prose[pos:m.start()])
        chunks.append(_fmt(val))
        pos = m.end()
    chunks.append(prose[pos:])
    return "".join(chunks)


def _story_by_beats(topic: str, facts: list[dict], angle: dict, need: set[float]) -> str:
    """Narrate beat-by-beat: one bounded call per beat (hook -> beats -> closer).

    A small local model compresses a whole-story ask into 1-2 sentences no
    matter how it is worded. Writing each beat as its OWN call - exactly the
    pattern that holds the line count in the theme stage - yields a
    deterministically long narration with each fact number demanded exactly
    once, so nothing repeats and nothing gets crammed.
    """
    labels = [b for b in (angle.get("beat_order") or []) if isinstance(b, str) and b.strip()]
    plan = ["HOOK"] + labels[:5] + ["CLOSER"]

    def _flag(first: bool, closer: bool) -> str:
        if first:
            return ("Make the hook sound like the opening scene of a war movie - one flat, "
                    "surprising image or number from the facts, never a greeting.")
        if closer:
            return ("Final line: exactly ONE earned question pointing back at the hook, or the "
                    "reveal made flat. Never a stack of questions.")
        return "No greeting. This is a mid-story beat - keep the tension rising, never a question."

    sentences = []
    prior: list[str] = []
    used: set[float] = set()
    for k, beat in enumerate(plan):
        is_hook = k == 0
        is_closer = k == len(plan) - 1
        unused = sorted(need - used)
        demand = ""
        if unused and not is_closer:
            demand = f"Include the exact figure {unused[0]:g} in this sentence, written as digits."
        if not demand:
            demand = "No specific figure required - use any fact number that fits."
        prompt = _render(
            STORY_BEAT_PROMPT,
            reversal_hint=angle.get("reversal") or "what the public believed was not what existed",
            fooled_hint=angle.get("fooled") or "the accepted story hid the real one",
            beat=beat,
            first_flag=_flag(is_hook, is_closer),
            facts=_facts_text(facts),
            prior="\n".join(prior) or "(this is the first sentence - nothing written yet)",
            demand=demand,
            closer_rule=_flag(False, is_closer),
        )
        out = _local(prompt, temperature=0.6, tag="story").strip()
        out = re.sub(r"^L?\d+[.:]\s*", "", out).strip()
        if not out:
            continue
        invented = [n for n in _canon_numbers(out) if n not in need and n != 1.0]
        if invented and not is_closer:
            nums = ", ".join(f"{n:g}" for n in invented[:3])
            print(f"    [story] beat '{beat}' invented {nums} - one number-safe retry")
            retry = _local(
                prompt + f"\n\nYou invented the figure(s) {nums} - they are NOT in the FACTS. "
                "Rewrite this beat with NO number except the ones in the FACTS, exact and written "
                "as digits.",
                temperature=0.5, tag="story",
            ).strip()
            retry = re.sub(r"^L?\d+[.:]\s*", "", retry).strip()
            if retry and not [n for n in _canon_numbers(retry) if n not in need and n != 1.0]:
                out = retry
        sentences.append(out)
        used |= set(_canon_numbers(out)) & need
        prior = (prior + [out])[-2:]
    return " ".join(sentences)


def write_story(topic: str, facts: list[dict]) -> str:
    print("\n    ANGLE (frame-lock)")
    angle = _patch_angle(_sanitize_angle(extract_angle(facts)))
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
    prose = _story_by_beats(topic, facts, angle, need)
    candidates.append(prose)
    short = _count_sentences(prose) < 8
    if any(b in prose for b in banned) or _example_leak(prose):
        print("    [story] echo of hint/example tokens detected - adding a clean re-roll")
        candidates.append(
            _local(
                base
                + "\n\nYou wrote the raw hint tokens ('vs', 'instead of reality', etc.) or copied an EXAMPLE "
                "phrase from the instructions wholesale. Rewrite: narrate THIS story from the facts alone, in "
                "your own invented words - none of the instruction's example wording is allowed here. "
                + tighten,
                temperature=0.6, tag="story",
            )
        )
        short = True
    while len(candidates) < 3 and short:
        best, n = max(_story_score(p, need) for p in candidates)
        top = max(candidates, key=lambda p: _story_score(p, need))
        if best >= 20 and 8 <= n <= 12 and not _example_leak(top):
            break
        print(f"    [story] best candidate scores {best} ({n} sentences) - one more attempt")
        candidates.append(_local(base + "\n\n" + tighten, temperature=0.6, tag="story"))
    # pick best-score candidate (ties broken by the size term inside _story_score)
    best, _ = max(_story_score(p, need) for p in candidates)
    prose = max(candidates, key=lambda p: _story_score(p, need))
    if _example_leak(prose):
        print("    [story] best still copies an example - dedicated anti-copy re-roll")
        prose = _local(
            base
            + "\n\nSTRICT FINAL CHECK: the narration still contains example phrases copied from the "
            "instructions - those exact words are forbidden here. Rewrite the SAME story using ONLY the "
            "given facts and your own fresh wording, every number as digits. " + tighten,
            temperature=0.6, tag="story",
        )
    n = _count_sentences(prose)
    print(f"    [story] selected best-of-{len(candidates)} (score {best}, {n} sentences)")
    if n < 8:
        prose = _local(
            base
            + f"\n\nYour previous narration had only {n} sentence(s). The minimum is 8. "
            "Write ONE sentence for EACH beat of your beat_order (hook, every beat, reveal, closer) "
            "- that alone is 6 to 9 sentences; add scenes for the remaining facts. Cover the facts "
            "one per sentence, in this order:\n" + _facts_text(facts)
            + "\nStill one sentence per line, still exact numbers written as digits."
            + " " + tighten,
            temperature=0.6, tag="story",
        )
        n = _count_sentences(prose)
    if n > 12:  # last resort: keep hook + first 11 sentences
        parts = [p for p in re.split(r"(?<=[.!?])\s+", prose.strip()) if p.strip()]
        print(f"    [story] engine-at-{n} - trimming deterministically to 12")
        prose = " ".join(parts[:12])
    # Guarantee fact figures read as digits in the final narration, regardless
    # of whether the model spelled them out ("seventy" -> "70", "two thousand"
    # -> "2,000"). Junk readings stay as written.
    return _digits_for_facts(prose, need)