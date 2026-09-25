"""Story agent - deterministic helpers (no model calls).

Everything in this module is pure and unit-testable: sentence shaping,
dedup, hook scoring, cold-open, compound splitting, capitalization, title-echo
stripping, and the microsecond deterministic fact-gate that catches invented
numbers and victories-out-of-defeat before any showrunner vote.
"""
from __future__ import annotations

import re

# ------------------------------------------------------------------ shaping

_MIN_RUN = 8  # verbatim token run shared with a kept line -> duplicate
_MAX_STORY_LINES = 12  # sweeter spot for one short video; hook + closer kept

# Instruction-tag/heading text and celebrity names the model sometimes copies
# verbatim from the injected speaking-style block instead of following it.
_STYLE_HEADINGS = re.compile(
    r"VOICE:|HOW HE TALKS|HOW TO OPEN|HOW TO CLOSE|CATCHPHRASE|WRITE THE ENTIRE STORY",
    re.I,
)
_CELEBRITY = re.compile(
    r"\b(schwarzenegger|donald trump|andrew tate)\b", re.I
)
# Imperative writing-directives in the style block. If the model pastes one of
# these as if it were narration, it is instruction text, not a story line. This
# is checked on EVERY line (including line 0) so a pasted "Open by..." opener is
# still caught even though line 0 skips the block-echo check below.
_STYLE_DIRECTIVE = re.compile(
    r"\bopen by\b|\bopen loop\b|\bnever start with\b|\bnever open with\b"
    r"|\bthe interrupt leads\b|\bthe provocation leads\b|\bthe boast-command leads\b"
    r"|\bthen hit them with\b|\bthen jump straight\b|\brank it against\b"
    r"|\bone sentence per line\b|\bwrite the entire story\b|\bnever copy a sentence\b"
    r"|\bthe only facts come\b|\bnever reuse a number\b|\bnever put the narrator\b"
    r"|\bnever add an outcome\b|\bnever type the voice\b|\bno filler\b"
    r"|\bno superlatives in every line\b|\bno phonetic\b|\bno movie one-liners\b"
    r"|\bno gym\b|\bno luxury\b|\bno self-credentialing\b|\bnot a cartoon villain\b"
    r"|\bnot only shouting\b|\bthe viewer must need\b|\bkeep an open loop\b"
    r"|\boutput only the story\b|\bmake it stop-scroll\b|\btell it with feeling\b"
    r"|\byou are not relaying\b|\blet that emotion bleed\b|\binclude technical elements only\b"
    r"|\bcut technical detail\b|\bshort but detailed\b|\bevery sentence must say\b"
    r"|\bbuild tension to a peak\b|\bput a stakes line\b|\bevery line should pull\b"
    r"|\bcut it or fold it\b|\bpick the concrete\b|\bleave it dangling\b"
    r"|\bpose the story itself\b|\bthe story is the hook\b|\bthe core shock\b"
    r"|\bget there in the very first line\b|\bno attention-grabbing boilerplate\b"
    r"|\bwrite it fresh for this story\b|\bthe scale gap\b|\bthe impossible odds\b",
    re.I,
)
# Story-scenario hooks: a question or contrast posed by the story itself, not
# an attention-grabber aimed at the viewer. Without these, _hook_score rates a
# good "how did X beat Y?" opener ~0 and _cold_open would demote it to a
# mid-story fact.
_HOOK_SCENARIO = re.compile(
    r"[?]$"
    r"|\b(how did|how could|how do|how does|what if|why would|what would)"
    r"|\b(can you imagine|ever wonder|imagine|what happens when|what happens next)"
    r"|\b(losing to|lost to|beaten by|defeated by|outnumbered|outgunned|took on|against all odds)\b",
    re.I,
)

_BLOCK_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _block_sentences(block: str) -> list[str]:
    """Instruction sentences of the speaking-style block (hints for echoing)."""
    return [s.strip() for s in _BLOCK_SENT_RE.split(block) if len(s.strip()) > 3]


def _tokens(line: str) -> list[str]:
    """All lowercase word tokens of a line, in order (punctuation kept apart)."""
    return re.findall(r"[a-z0-9']+", line.lower())


_ABBR = frozenset({
    "mr", "mrs", "ms", "dr", "st", "mt", "jr", "sr", "vs", "etc", "e.g",
    "i.e", "u.s", "u.k", "no", "fig", "dept", "col", "gen", "lt", "capt",
    "maj", "sgt",
})
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _to_sentences(text: str) -> list[str]:
    """Normalize arbitrary model output into one-sentence-per-line.

    The model often returns the story as one long paragraph despite the
    prompt; splitting deterministically keeps the dedup guard effective and
    gives the agent clean per-sentence input.
    """
    raw = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not raw:
        return []
    chunks: list[str] = []
    for ln in raw:
        parts = _SENT_SPLIT.split(ln)
        buf = ""
        for part in parts:
            head = part.split(" ", 1)[0].rstrip(".").lower()
            if buf and head in _ABBR:
                buf += " " + part
                continue
            if buf:
                chunks.append(buf)
            buf = part
        if buf:
            chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


def _longest_run(a: list[str], b: list[str]) -> int:
    """Longest contiguous common token sequence (proper LCS-substring DP)."""
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _dedupe_story(story: str) -> str:
    """Drop lines that echo a previously-kept line.

    Catches both whole-line repeats and verbatim source clauses reused inside
    a longer sentence (what llama3.2 does: it stitches the same phrase into
    two different lines). Matches raw contiguous token runs so a long shared
    clause is caught even when the sentences around it differ.
    """
    lines = _to_sentences(story)
    if not lines:
        return ""
    kept: list[str] = []
    kept_tokens: list[list[str]] = []
    for line in lines:
        toks = _tokens(line)
        if not toks:
            kept.append(line)
            continue
        dup = any(
            _longest_run(toks, prev) >= _MIN_RUN for prev in kept_tokens
        )
        if not dup:
            kept.append(line)
            kept_tokens.append(toks)
    return "\n".join(kept)


def _style_leak(line: str, block: str) -> bool:
    """True if a story line echoes the injected speaking-style block.

    Small models sometimes paste the persona section verbatim ("VOICE:
    ARNOLD SCHWARZENEGGER - a battle-hardened commander...") into the story
    instead of following it. Nothing that quotes an instruction heading, names
    the narrator, or copies an instruction sentence is ever a real narration
    line.
    """
    if not line:
        return True
    if _STYLE_HEADINGS.search(line) or _CELEBRITY.search(line):
        return True
    if _STYLE_DIRECTIVE.search(line):
        return True
    toks = _tokens(line)
    for s in _block_sentences(block):
        run = _longest_run(toks, _tokens(s))
        # Long verbatim copy, OR a short sentence that is ~a full echo of an
        # instruction sentence (e.g. "Warm and encouraging, never cruel.").
        if run >= _MIN_RUN or (run >= 4 and run >= 0.8 * len(toks)):
            return True
    return False


def _merge_fragments(lines: list[str]) -> list[str]:
    """Fold ultra-short lines into the previous one.

    The model often ends a beat with a catchphrase tag ("It's true. Very
    true.") and the sentence splitter turns it into two standalone 2-token
    lines. As separate script lines those become 1-second dead air with
    hallucinated queries - attaching them to the line they confirm is the
    right shape. A question fragment ("with what?") is NOT merged: it is a
    real beat and capitalizes fine on its own line.
    """
    out: list[str] = []
    for line in lines:
        if out and len(_tokens(line)) < 4 and not line.rstrip().endswith("?"):
            out[-1] = f"{out[-1]} {line}".strip()
        else:
            out.append(line)
    return out


def _cap_story(lines: list[str]) -> list[str]:
    """Hard cap the story at _MAX_STORY_LINES keeping the hook and the closer.

    The middle is sampled evenly so a bloated 36-line output collapses to a
    tight 12 while the opening beat and the final line survive.
    """
    if len(lines) <= _MAX_STORY_LINES:
        return lines
    keep = [lines[0]]
    mid = lines[1:-1]
    if mid:
        budget = _MAX_STORY_LINES - 2
        if len(mid) <= budget:
            keep += mid
        else:
            step = (len(mid) - 1) / (budget - 1) if budget > 1 else 0
            idx = sorted({round(i * step) for i in range(budget)})
            keep += [mid[i] for i in idx]
    keep.append(lines[-1])
    return keep[:_MAX_STORY_LINES]


# Words that scream "stop and look" when they sit inside a real fact: concrete
# numbers, absolutes, doom/stakes, deception, survival. Empty words (names,
# places, institutions, adjectives) get nothing - they are not hooks.
_HOOK_PAYLOAD = (
    "secret", "secrets", "faked", "fake", "deception", "decoy", "phantom",
    "doomed", "suicide", "survival", "survived", "survive", "slaughter",
    "killed", "killing", "murder", "died", "death", "dead", "torture",
    "trapped", "impossible", "never", "nobody", "nothing", "everything",
    "alone", "only", "wait", "longest", "largest", "biggest", "million",
    "thousands", "hundreds", "tiny", "daily", "human", "last", "first",
    # identity/drama - a person or stakes you can picture
    "son", "sons", "daughter", "daughters", "slave", "widow", "orphan",
    "prisoner", "refugee", "survivor", "traitor", "betrayal", "betrayed",
    "revolt", "rebel", "uprising", "mutiny", "impostor", "disaster",
)
_FORBIDDEN_OPENER = re.compile(
    r"^(in|by|during|on|after|before|since|at|from)\s+(the\s+|even\s+|early\s+|late\s+)?\d"
    r"|^(on|in|by|during|at|from)\s+(january|february|march|april|may|june|july|"
    r"august|september|october|november|december)\s+\d"
    r"|^(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+\d"
    r"|^\d{4}\b|^(this is a story about|the history of|let me tell you)"
    r"|^(even worse|worse|but wait|and then|like i said|this one takes the cake)"
    r"|^(incredibly|unbelievably|amazingly)\b"
    r"|\b(nazi|german|soviet|american|british|french|polish|japan|japanese|russian|italian)\b\s+"
    r"(\w+\s+){0,3}(was|were|had|aimed|began|invaded|attacked)",
    re.I,
)


def _hook_score(line: str) -> int:
    """A painless deterministic proxy for 'would this stop the thumb?'."""
    toks = _tokens(line)
    if not toks:
        return -10
    score = 0
    for w in toks:
        base = w.rstrip("s")
        if w in _HOOK_PAYLOAD or base in _HOOK_PAYLOAD:
            score += 2
    for num in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", line):
        n = float(num.replace(",", ""))
        if 1000 <= n <= 2100:
            continue  # a bare year is context, not a hook
        score += 3
        if n >= 100000:
            score += 2
    score += sum(2 for p in ("…", "?!", "...") if p in line)
    # A story-scenario question ("how did X beat Y?") IS the hook - never let
    # _cold_open demote it to a mid-story fact.
    if _HOOK_SCENARIO.search(line):
        score += 8
    # penalize forbidden textbook openers hard - they are the exact thing a
    # stop-scroll video must never lead with.
    if _FORBIDDEN_OPENER.search(line):
        score -= 8
    # very short or over-stuffed sentences rarely make good hooks either.
    if len(toks) < 5:
        score -= 3
    if len(toks) > 24:
        score -= 2
    return score


def _cold_open(lines: list[str]) -> list[str]:
    """Guarantee line 1 is the hungriest line (ONLY line 1 - the rest keep
    their order and arc).

    The hook rule applies only to the first 1-2 sentences: that one line must
    stop a thumb mid-scroll. If the model opened with a setup/forbidden
    formula, move the single highest-scoring sentence to the front as a cold
    open, then let the story resume where it left off. A line 1 that already
    carries enough tension (>= _HOOK_GOOD) is never touched, so a decent
    model-written hook is preserved.
    """
    _HOOK_GOOD = 4
    if len(lines) < 2:
        return lines
    if _hook_score(lines[0]) >= _HOOK_GOOD:
        return lines
    best = max(range(1, len(lines)), key=lambda i: _hook_score(lines[i]))
    # no strong candidate, or the current line 1 is close enough - keep order.
    if _hook_score(lines[best]) <= 0:
        return lines
    if _hook_score(lines[best]) <= _hook_score(lines[0]) + 3:
        return lines
    return [lines[best]] + lines[:best] + lines[best + 1:]


# A dash/semicolon that glues two thoughts ("Stop scrolling, listen to me -
# 1933 was the year...") is a line break, not punctuation. Small models love
# it; the AGENT is told never to split a sentence on a dash, so nothing ever
# separates it. Split it here, deterministically, into two lines.
_COMPOUND_SEP = re.compile(r"\s+[-–—]{1,2}\s+|;\s+")
_CLAUSE_START = re.compile(
    r"^(he|she|it|they|we|you|i|this|that|these|those|there|then|but|so|and|"
    r"when|while|after|before|by|in|during|on|at|from|the|a|an|"
    r"no|not|never|every|most|many|some|\d)\b",
    re.I,
)


def _split_compounds(lines: list[str]) -> list[str]:
    """Break a dash/semicolon-joined compound into separate lines.

    Two different cases:
    - A SEMICOLON joins two independent clauses, full stop. Split when both
      halves are substantial (>=4 tokens, so no fragments) - no other reason
      to keep it, and the AGENT is told never to split on a semicolon, so
      nothing downstream would ever separate it.
    - A DASH only splits when the left half is a scroll-stop hook, or a long
      line whose right half starts a fresh clause. A dash used for apposition
      ("...culminated in the Holocaust - six million killed") stays intact.
    """
    out: list[str] = []
    for ln in lines:
        parts = [p.strip() for p in _COMPOUND_SEP.split(ln)]
        if len(parts) <= 1:
            out.append(ln)
            continue
        is_glue_semicolon = bool(re.search(r";\s+", ln))
        if is_glue_semicolon and all(len(_tokens(p)) >= 4 for p in parts):
            out.extend(parts)
        elif (
            all(len(_tokens(p)) >= 4 for p in parts)
            and (
                _HOOK_SCENARIO.search(parts[0])
                or (len(_tokens(ln)) > 24 and _CLAUSE_START.match(parts[-1]))
            )
        ):
            out.extend(parts)
        else:
            out.append(ln)
    return out


def _capitalize_lines(lines: list[str]) -> list[str]:
    """Restore a capital letter at the start of every line.

    Small models frequently start a line lowercase (or a merged fragment that
    began mid-sentence). Every line is its own spoken sentence, so it must
    begin capitalized - deterministically, safe for TTS, never touching the
    rest of the line.
    """
    out = []
    for ln in lines:
        m = re.search(r"[A-Za-z0-9]", ln)
        if m and ln[m.start()].islower():
            ln = ln[:m.start()] + ln[m.start()].upper() + ln[m.start() + 1:]
        out.append(ln)
    return out


def _strip_title_echo(line: str, topic: str) -> str:
    """Cut a leading copy of the topic title off a story line.

    The article body often starts with the repeated H1 + the lede, and small
    models reproduce that lead-in verbatim ("Deadly Medicine: Creating the
    Master Race began in 1933..."). The topic title is not narration - strip
    it so the line becomes the actual opener (or "" if the whole line is just
    the title echo).
    """
    title = re.sub(r"^\s*\d+[\s.:-]+", "", topic or "").lower()
    t = _tokens(title)
    toks = _tokens(line)
    if not t or not toks:
        return line
    # match as many title words as the line actually opens with
    n = 0
    for a, b in zip(t, toks):
        if a != b:
            break
        n += 1
    # only treat it as a title echo when a real chunk led (>=4 words)
    if n < 4:
        return line
    # cut the matched words (in order, from the line's start) out of the
    # ORIGINAL line, keeping its case: ^ any junk, then the n words, then rest.
    words = [re.escape(w) for w in toks[:n]]
    pattern = r"^\W*" + r"\W+".join(words)
    m = re.match(pattern, line, re.I)
    if not m:
        return line
    rest = line[m.end():].strip(" :,.;-–")
    # a strip that leaves nothing to narrate (a bare "France." tail) means the
    # sentence just re-titled itself - keep the whole line rather than emit a
    # 1-word fragment.
    if len(_tokens(rest)) < 3:
        return line
    return rest


def _strip_leading_dash(line: str) -> str:
    return re.sub(r"^\s*[-–—]{1,2}\s+", "", line)


# ------------------------------------------------- deterministic fact-gate

_NUM_TOKEN = re.compile(r"\b\d[\d,]*\b(?:\.[\d]+)?")
# Spelled-out number words. Small models invent round crowd figures
# ("forty-five thousand veterans") and rephrase the material's digits as
# words; both must resolve to the same VALUE, not the same spelling.
_DIGIT_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000,
    "billion": 1_000_000_000,
}


def _parse_spelled(run: list[str]) -> int | None:
    """Parse a contiguous run of number words into ONE value.

    "forty five thousand" -> 45000, "two thousand eight hundred" -> 2800,
    "a million" -> 1_000_000 (the 'a' is skipped by the tokenizer). Falls back
    to None for anything unparseable instead of guessing.
    """
    total, current = 0, 0
    for tok in run:
        v = _DIGIT_WORDS.get(tok)
        if v is None:
            return None
        if v >= 100 and current == 0:
            total += v
        elif v >= 100:
            total += current * v
            current = 0
        else:
            current += v
    return total + current


def _stack_values(text: str) -> set[int]:
    """All number VALUES in a text, in any spelling (digits or words)."""
    toks = _tokens(text)
    values: set[int] = set()
    # contiguous word-number runs (also links spelled pairs like "forty five")
    i = 0
    n = len(toks)
    while i < n:
        if toks[i] in _DIGIT_WORDS:
            run_start = i
            while i < n and toks[i] in _DIGIT_WORDS:
                i += 1
            v = _parse_spelled(toks[run_start:i])
            if v is not None:
                values.add(v)
            continue
        i += 1
    for m in _NUM_TOKEN.finditer(text):
        num = m.group(0).replace(",", "")
        try:
            values.add(int(float(num)))
        except ValueError:
            pass
    return values


def _brief_fact_terms(brief: dict) -> str:
    parts = [
        brief.get("incident") or "",
        brief.get("person") or "",
        brief.get("stakes") or "",
        brief.get("detail") or "",
        brief.get("outcome") or "",
        brief.get("who_else") or "",
        brief.get("numbers") or "",
    ]
    return " ".join(parts)


def _outcome_blockers(brief: dict, draft: str) -> list[str]:
    """If the scene's true ending is a failure/disband, the story must not
    claim a triumph or a clean win. Driven by the OUTCOME field, which the
    scene picker is required to extract from the material verbatim."""
    outcome = (brief.get("outcome") or "").lower()
    if not outcome:
        return []
    d = draft.lower()
    fail_markers = ("failed", "ordered to dissolve", "dissolve", "retreat",
                    "retreated", "withdraw", "withdrew", "abandoned")
    # Deliberately phrase-level, not bare words: "no hope of victory" is an
    # honest line about a lost cause and must NOT be flagged, while "victory
    # was theirs" (a claimed win) must be.
    win_markers = ("they broke through", "she broke through", "broke through the",
                   "victory was theirs", "they won", "won against", "won the day",
                   "won the battle", "erased an entire", "silenced the guns",
                   "a stunning victory", "the attack succeeded",
                   "the attack was a success", "they defeated",
                   "they reached the bridgehead and held it")
    if any(m in outcome for m in fail_markers):
        hits = sorted({m for m in win_markers if m in d}, key=len, reverse=True)
        if hits:
            return [
                f"THE SCENE ENDED IN {outcome.upper()} - the story must end in that "
                f"same failure. Remove any implication of victory or a broken "
                f"through: {', '.join(hits)}."
            ]
    return []


def _numeric_figures(problems: list, draft: str, brief_values: set) -> None:
    """Every numeric figure a draft names must have a value in the material."""
    for m in _NUM_TOKEN.finditer(draft.lower()):
        num = m.group(0)
        try:
            value = int(float(num.replace(",", "")))
        except ValueError:
            continue
        if value not in brief_values:
            problems.append(
                f"The number {num} is not in the incident material. Replace it "
                f"with the exact figure the material gives (or drop it)."
            )


def _spelled_figures(problems: list, draft: str, brief_values: set) -> None:
    """Spelled-out magnitude runs (e.g. \"forty-five thousand\") not in material.

    Only runs containing a magnitude word (hundred/thousand/million) are
    evaluated, so ordinary speech like \"the first grenadier\" is never flagged
    but invention like \"forty-five thousand veterans\" is caught.
    """
    toks = _tokens(draft.lower())
    i, n = 0, len(toks)
    while i < n:
        if toks[i] in _DIGIT_WORDS and not re.fullmatch(
            r"one|two|three|four|five", toks[i]
        ):
            run_start = i
            while i < n and toks[i] in _DIGIT_WORDS:
                i += 1
            run = toks[run_start:i]
            has_magnitude = any(
                t in ("hundred", "thousand", "million", "billion") for t in run
            )
            if has_magnitude:
                value = _parse_spelled(run)
                phrase = " ".join(run)
                if value is not None and value not in brief_values:
                    problems.append(
                        f"You named a count the material never states "
                        f"(\"{phrase}\"). Use the material's exact figures only."
                    )
            continue
        i += 1


def _fact_gate(draft: str, brief: dict) -> list[str]:
    """Return concrete fact violations in a draft ([] = clean).

    Numbers a draft names must resolve to a VALUE the material stated (in any
    spelling: "50" or "fifty"). Pure prose is never touched, so a
    stylistically different but factually honest rewrite passes.
    """
    problems: list[str] = []
    facts = _brief_fact_terms(brief)

    # Outcome wording must respect the true ending.
    problems += _outcome_blockers(brief, draft)

    # Numbers: every figure the story names must have a value present in the
    # material. Round crowd figures the model invents ("45,000", "six tanks")
    # are not in the brief and get flagged.
    brief_values = _stack_values(facts)
    _numeric_figures(problems, draft, brief_values)

    # Spelled-out figures: scan contiguous number-word runs (see helper).
    _spelled_figures(problems, draft, brief_values)
    return problems


# ------------------------------------------------------------ post-process

def _postprocess(text: str, block: str, topic: str) -> str:
    """Deterministic shaping shared by every draft (initial + refinements).

    Keeps at most 12 lines with the highest-scoring line forced to the front as
    a cold open, splits dash/semicolon-joined compounds, merges fragments, and
    restores capitals. Raises ValueError when the draft is nothing but style
    instructions or a title echo, so a wedged reply cannot flow on.
    """
    lines = [
        _strip_leading_dash(ln)
        for ln in _to_sentences(text)
        if not _style_leak(ln.strip(), block)
    ]
    if not lines:
        raise ValueError("story stage returned only style instructions - cannot continue")
    if topic and lines:
        head = lines[:3]
        kept_head = [_strip_title_echo(ln, topic) for ln in head]
        kept_head = [s for s in kept_head if s]
        lines = kept_head + lines[3:]
        if not lines:
            raise ValueError("story stage was only a title echo - cannot continue")
    lines = _to_sentences(_dedupe_story("\n".join(lines)))
    lines = [ln for ln in lines if ln]
    lines = _split_compounds(lines)
    lines = _merge_fragments(lines)
    lines = _cap_story(lines)
    lines = _cold_open(lines)
    lines = _capitalize_lines(lines)
    return "\n".join(lines)


def _fact_notes(problems: list[str]) -> str:
    return (
        "DETERMINISTIC FACT-CHECK - these are hard, engineering-level "
        "blocks, not taste:\n"
        + "\n".join(f"- {p}" for p in problems)
    )


def _story_quality(draft: str) -> float:
    """A cheap deterministic proxy for 'is this draft worth shipping?'.

    A rewrite can be WORSE than the draft before it (one real run: the model
    rewrote a story and flattened a strong opener into "On June 17th, the 1st
    Grenadier Division faced..."). The loop keeps the best fact-clean draft it
    has seen, not the last one. Hook presence dominates - a flat fact opener
    loses hard against any story-scenario opener.
    """
    lines = _to_sentences(draft)
    if not lines:
        return -999.0
    score = 0.0
    # A dead flat opener is disqualifying unless nothing better exists.
    if _hook_score(lines[0]) >= 4:
        score += 40
    elif _hook_score(lines[0]) <= 0:
        score -= 20
    # substance: we want a story with an arc, not a two-liner or a wall
    score += min(len(lines), _MAX_STORY_LINES) * 2
    words = sum(len(_tokens(ln)) for ln in lines)
    if words < 60:
        score -= 10
    elif words > 240:
        score -= 4
    return score


def _pick_best(a: str, b: str) -> str:
    """The better of two fact-clean drafts, by `_story_quality`."""
    return a if _story_quality(a) >= _story_quality(b) else b