"""Stage 1 - STORY: raw source text -> a short, detailed, easy-to-follow story,
written directly in the narrator's speaking style in ONE pass.

One narrow LLM call. The prompt injects a speaking style (Arnold/Trump/Tate/
narrator) so the whole story comes out in the character's voice, keeping
technical detail ONLY where the story needs it and staying natural, adult-
friendly prose (not baby talk, not a dry report). No separate theme stage.

The output passes through a small deterministic dedup guard: small local
models (llama3.2 etc.) sometimes echo a source clause or repeat a phrase, and
that must not survive into the narration.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from src.config.script_lab_prompts import STORY_PROMPT
from src.config.speaking_styles import get_speaking_style
from src.services.script_lab.llm import _local
from src.services.script_lab.text import _render

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
    r"|\bcut it or fold it\b|\bpick the concrete\b|\bleave it dangling\b",
    re.I,
)
# Direct-address scroll-stoppers. These ARE the hook we now want: a persona
# interrupting the viewer's thumb. Without this, _hook_score rates them ~0
# (no numbers/payload) and _cold_open would demote them to a mid-story fact.
_HOOK_INTERRUPT = re.compile(
    r"\b(stop scrolling|stop the scroll|listen to me|listen close|listen up)"
    r"|\b(hey you|hey - you|right there|hey, you)"
    r"|\b(ever heard|you ever|about to tell you|about to tell|did you ever)"
    r"|\b(blow your mind|blow your candle|doomscroller|you have no idea)"
    r"|\b(this story will|stop scrolling your finger|hold on)\b",
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


def _style_leak(line: str, block: str, check_echo: bool = True) -> bool:
    """True if a story line echoes the injected speaking-style block.

    Llama small models sometimes paste the persona section verbatim ("VOICE:
    ARNOLD SCHWARZENEGGER - a battle-hardened commander reviewing the
    operation. Warm and encouraging, never cruel.") into the story instead of
    following it. Nothing that quotes an instruction heading, names the
    narrator, or copies an instruction sentence is ever a real narration line.

    `check_echo=False` skips the instruction-echo pass. That is used for line 0
    only: the opener is SUPPOSED to echo the persona's scroll-stop phrase, and
    the example phrases live in the block, so echoing them on line 1 is the
    hook, not a leak. Headings / celebrity names / directives are still checked
    on line 0 so a pasted "VOICE:" or "Open by..." opener is still dropped.
    """
    if not line:
        return True
    if _STYLE_HEADINGS.search(line) or _CELEBRITY.search(line):
        return True
    if _STYLE_DIRECTIVE.search(line):
        return True
    if not check_echo:
        return False
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
    right shape.
    """
    out: list[str] = []
    for line in lines:
        if out and len(_tokens(line)) < 4:
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
    r"|^\d{4}\b|^(this is a story about|the history of|let me tell you)"
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
    # A persona scroll-stopper IS the hook - never let _cold_open demote it.
    if _HOOK_INTERRUPT.search(line):
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

    Only fires when both halves are substantial (>=4 tokens, so no fragments)
    AND the split is real: the left half is a scroll-stop hook, or a long line
    whose right half starts a fresh clause. A dash used for apposition
    ("...culminated in the Holocaust - six million killed") stays intact.
    """
    out: list[str] = []
    for ln in lines:
        parts = [p.strip() for p in _COMPOUND_SEP.split(ln)]
        if (
            len(parts) > 1
            and all(len(_tokens(p)) >= 4 for p in parts)
            and (
                _HOOK_INTERRUPT.search(parts[0])
                or (len(_tokens(ln)) > 24 and _CLAUSE_START.match(parts[-1]))
            )
        ):
            out.extend(parts)
        else:
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


def build_story(story: str, style: str = "narrator", topic: str = "") -> str:
    """Turn `story` into a short, detailed, easy-to-follow narrative.

    `style` selects the speaking style injected into the prompt (narrator,
    arnold, trump, andrew_tate). `topic` (the topic title) is used to strip a
    leading title echo that small models copy from the article's H1. Returns
    the story as a string (one sentence per line). Raises on empty model
    output so a wedged reply cannot flow on.
    """
    block = get_speaking_style(style)
    out = _local(
        _render(STORY_PROMPT, style=block, story=story),
        temperature=0.4,
        tag="story",
    )
    text = (out or "").strip()
    if not text:
        raise ValueError("story stage returned nothing - cannot continue")
    lines = [
        ln for i, ln in enumerate(_to_sentences(text))
        if not _style_leak(ln.strip(), block, check_echo=(i != 0))
    ]
    if not lines:
        raise ValueError("story stage returned only style instructions - cannot continue")
    if topic and lines:
        # The article H1 echo can land on any of the first lines: the hook now
        # owns line 0, so the title often surfaces on line 1-2 instead. Strip it
        # from the first few lines and drop any line that was purely the title.
        head = lines[:3]
        kept_head = [_strip_title_echo(ln, topic) for ln in head]
        kept_head = [s for s in kept_head if s]
        lines = kept_head + lines[3:]
        if not lines:  # extreme: every head line was a title echo
            raise ValueError("story stage was only a title echo - cannot continue")
    lines = _to_sentences(_dedupe_story("\n".join(lines)))
    lines = [ln for ln in lines if ln]
    lines = _split_compounds(lines)
    lines = _merge_fragments(lines)
    lines = _cap_story(lines)
    lines = _cold_open(lines)
    return "\n".join(lines)