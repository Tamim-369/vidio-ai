"""Number fidelity engine (pure, model-free, unit-testable).

Every stage that re-writes facts into narration depends on canonical number
extraction: `_canon_numbers` turns digits, spelled forms, years, short years,
and fractions into a flat list of semantic values. The audit helpers build on
it so a re-voice is only accepted if the factual numbers survived unchanged -
number fidelity outranks added flavor, always.
"""
from __future__ import annotations

import re

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000,
}

# matches spelled numbers that can appear WITHOUT a multiplier (the old parser
# only caught "... hundred/thousand/...", so "sixty-three miles" and
# "nineteen ninety six" were invisible to the audit - the corruption pass bug).
_SPELLED_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion)\b(?:[\s-]+\b(?:and\b[\s-]+\b)?"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion)\b)*",
    re.IGNORECASE,
)

# "--twenty --thirty" group used to detect year-style readings like
# "nineteen ninety six" (1996), which the naive sum parser would read as 115.
_TENS_PAT = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
_UNIT_PAT = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
             r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)")
_TEEN_PAT = (r"(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
             r"eighteen|nineteen)")
# year-style readings: "nineteen ninety six", "twenty twenty", "nineteen sixty",
# "sixty three" (plain compound). Guarded so a trailing multiplier ("fifty five
# thousand") is NOT split off as a year - it stays one number read by the spelled
# pass. "one hundred" never matches because "one" is not a teen/tens word.
_YEAR_RE = re.compile(
    rf"\b(?:{_TEEN_PAT}|{_TENS_PAT})"
    rf"(?:(?:[\s-]+(?:{_TENS_PAT})(?:[\s-]+{_UNIT_PAT})?)|(?:[\s-]+{_UNIT_PAT}))"
    rf"(?![\s-]+(?:(?:hundred|thousand|million|billion))\b)"
)

# short-year forms like "'96", "'44" (a 20th-century story) map to 19YY so a
# narration that writes "in '96" still matches the fact value 1996 instead of
# being read as a bare 2-digit number and flagging a corruption.
_SHORT_YEAR_RE = re.compile(r"(?<![a-zA-Z0-9])'\d{2}\b")

# fractions and loose quantities the story writer uses to dodge an exact fact
# figure ("close to 500 yards" -> "half-a-mile"). Parse them as decimal values
# so a drift from the factual number triggers the audit instead of passing as
# ink. Only standalone fraction words - never inside "halfway"/"et cetera".
_FRACTION_WORDS = {
    "half": 0.5, "one-half": 0.5, "a-half": 0.5,
    "quarter": 0.25, "a-quarter": 0.25, "one-quarter": 0.25,
    "three-quarters": 0.75, "three-quarter": 0.75,
}


def _eval_spelled(words: list[str]) -> float:
    total = 0.0
    cur = 0.0
    for w in words:
        if w == "and":
            continue
        v = _NUM_WORDS[w]
        if v < 100:
            cur += v
        elif v == 100:
            cur = (cur or 1.0) * 100
        else:  # thousand / million / billion: scale everything so far, then emit
            total += (cur or 1.0) * v
            cur = 0.0
    return total + cur


def _canon_numbers(text: str) -> list[float]:
    """Extract the semantic values of every number in a sentence as floats.

    Handles digits ("30,000", "500", "3.5 million") plus spelled forms with or
    without multipliers ("sixty-three", "five hundred", "a hundred and six") and
    year-style readings ("nineteen ninety six" -> 1996). "one"/"a" article
    fillers are ignored so article-vs-number rewording ("a night" vs "one
    night") never fails the fidelity audit.
    """
    out: list[float] = []
    work = text
    for m in _SHORT_YEAR_RE.finditer(text):
        out.append(1900 + float(m.group(0).strip("'")))
        work = work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]
    for w in sorted(_FRACTION_WORDS, key=len, reverse=True):
        pat = re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE)
        n = len(pat.findall(work))
        if n:
            out.extend([float(_FRACTION_WORDS[w])] * n)
            work = pat.sub(" " * len(w), work)
    # digits, optionally scaled by a trailing multiplier word: "3.5 million",
    # "100 thousand". The matched region is blanked so the spelled pass below
    # does not also tally the bare multiplier word.
    for m in re.finditer(
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\s+(?:million|billion|thousand|hundred)\b",
        text, re.IGNORECASE,
    ):
        num, mult = m.group(0).strip().split()
        val = float(re.sub(r"[^\d.]", "", num)) * _NUM_WORDS[mult.lower()]
        out.append(val)
        work = work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]
    for m in re.finditer(r"(?<![A-Za-z0-9])'\d{2}\b", work):
        out.append(1900 + int(m.group(0).strip("'")))
        work = work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]
    for m in re.finditer(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", work):
        out.append(float(re.sub(r"[^\d.]", "", m.group(0))))
    for m in _YEAR_RE.finditer(text):
        toks = re.split(r"[\s-]+", m.group(0).strip())
        first = _NUM_WORDS.get(toks[0].lower())
        second = _NUM_WORDS.get(toks[1].lower()) if len(toks) > 1 else 0
        third = _NUM_WORDS.get(toks[2].lower()) if len(toks) > 2 else 0
        if first is not None and first >= 10 and second is not None and second >= 20:
            val = first * 100 + second + third
        else:
            val = sum(v for v in (first, second, third) if v is not None)
        out.append(float(val))
        work = work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]
    for m in _SPELLED_RE.finditer(work):
        words = re.split(r"[\s-]+", m.group(0).strip())
        w0 = _NUM_WORDS.get(words[0].lower()) if words else None
        if w0 is None:
            continue
        # a bare multiplier alone ("... and million more", "3.5 million" ->
        # blanked above) is not a number; but "hundred and six" / "thousand
        # five hundred" are. A run that starts on a multiplier must therefore
        # contain at least two number words to be a real reading.
        if w0 >= 100 and len(words) == 1:
            continue
        out.append(float(_eval_spelled([w.lower() for w in words])))
    return [round(n, 3) for n in out if n and n != 1.0]


def _theme_need(orig: str, fact_need: set[float]) -> list[float]:
    """Which numbers in an original sentence MUST survive a re-voice.

    Only numbers that carry factual weight are enforced: those that match the
    facts, or any 3+ digit value (years, 30,000, etc.). Incidental readings
    like "World War Two" -> 2.0 or "the 1980s" -> 1980 are rehearsal words, not
    facts, and must NOT force a voice fallback - otherwise trump saying "WWII"
    is rejected for "losing" a 2.0 the sentence never meant as a number.
    """
    return [n for n in _canon_numbers(orig) if n in fact_need or n >= 100]


def _numbers_survived(orig: str, re_voiced: str, fact_need: set[float]) -> bool:
    need = _theme_need(orig, fact_need)
    have = set(_canon_numbers(re_voiced))
    if not need:
        return True
    return all(n in have for n in need) and all(h in set(need) for h in have)


def _fact_numbers(facts: list[dict]) -> list[float]:
    out: list[float] = []
    for f in facts:
        for k in ("value", "fact", "name"):
            s = f.get(k)
            if isinstance(s, str) and s:
                out += _canon_numbers(s)
    return out