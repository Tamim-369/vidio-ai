"""Stage 3 - THEME: prose -> prose in a persona voice (arnold/trump/narrator).

Re-voices the narration line by line so a small model can hold the voice
without drifting: each original sentence is re-voiced in one bounded call.
Every number in the original must survive verbatim or the original line is
kept - number fidelity outranks added flavor, always. A theme not in THEMES
(e.g. the plain narrator style) passes the prose through unchanged.
"""
from __future__ import annotations

import re

from src.config.script_lab_prompts import THEME_LINE_PROMPT, THEMES
from src.services.script_lab.facts import _facts_block
from src.services.script_lab.llm import _local
from src.services.script_lab.numbers import (
    _canon_numbers,
    _fact_numbers,
    _numbers_survived,
    _theme_need,
)
from src.services.script_lab.text import _count_sentences, _render


def _shingles(text: str) -> set:
    """Lowercased word bigrams + trigrams: the unit we compare for 'does this
    line repeat an earlier line's phrasing?' 2+3-grams catch short recycled
    chunks ("over two thousand", "a huge failure") in otherwise diverging lines.
    """
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    if len(words) < 2:
        return set(words)
    bigrams = {" ".join(t) for t in zip(words, words[1:])}
    if len(words) < 3:
        return bigrams
    trigrams = {" ".join(t) for t in zip(words, words[1:], words[2:])}
    return bigrams | trigrams


def _echo_overlap(line: str, prior: list) -> float:
    """Best Jaccard-like trigram overlap of `line` against any already-voiced
    line (normalized by the shorter set). 0.0 if there is nothing to compare.
    """
    if not prior:
        return 0.0
    a = _shingles(line)
    if not a:
        return 0.0
    best = 0.0
    for p in prior:
        b = _shingles(p)
        if not b:
            continue
        best = max(best, len(a & b) / min(len(a), len(b)))
    return best


def apply_theme(prose: str, theme: str, facts: list[dict] | None = None) -> str:
    if theme not in THEMES:
        return prose
    moves = THEMES[theme]
    facts_block = _facts_block(facts) if facts else "(no fact anchor provided - use only the narration)"
    fact_need = set(_fact_numbers(facts)) if facts else set()
    sentences = [p for p in re.split(r"(?<=[.!?])\s+", prose.strip()) if p.strip()]
    re_voiced = []
    for i, s in enumerate(sentences):
        first = "A one-word greeting like 'Hey.' is allowed ONLY fused in front of this first sentence. Make the hook sound like the opening scene of a war movie." if i == 0 else "No greeting."
        prompt = _render(
            THEME_LINE_PROMPT,
            moves=moves,
            prose=prose,
            sentence=s,
            first_flag=first,
        )
        if not _numbers_survived(s, s, fact_need):
            print(f"    [theme] line {i+1} original itself fails the number audit ({_canon_numbers(s)} vs facts {sorted(fact_need)}) - keeping original anyway")
        out = _local(prompt, temperature=0.4, tag="theme", repeat_penalty=1.3).strip()
        if not out or not _numbers_survived(s, out, fact_need):
            important = ", ".join(f"{n:g}" for n in _theme_need(s, fact_need))
            keep = f" You MUST keep every important number of the original sentence EXACTLY: {important}." if important else ""
            retry = _local(
                prompt + "\n\nYour re-voice must keep every important number from the original. " + keep,
                temperature=0.25, tag="theme", repeat_penalty=1.3,
            ).strip()
            if retry and _numbers_survived(s, retry, fact_need):
                out = retry
            else:
                print(f"    [theme] line {i+1} dropped/kept-original (number check) - voice fallback")
                line = s
                re_voiced.append(line)
                continue
        frag = re.split(r"(?<=[.!?])\s+", out.strip())
        if len(frag) > 2:  # run-on guard: 2 sentences is fine, 3+ is a ramble
            print(f"    [theme] line {i+1} emitted {len(frag)} sentences - concision retry")
            tight = _local(
                prompt
                + "\n\nTwo sentences maximum. Cut the rest. Keep the key facts and the voice.",
                temperature=0.25, tag="theme", repeat_penalty=1.3,
            ).strip()
            if tight and _numbers_survived(s, tight, fact_need):
                out = tight
            else:  # tight retry lost the number - keep the (truncated) rambler
                frag = re.split(r"(?<=[.!?])\s+", out.strip())
                out = " ".join(frag[:2])
        line = out.strip()
        if _echo_overlap(line, re_voiced) >= 0.4:
            print(f"    [theme] line {i+1} repeats an earlier line's phrasing - dedup retry")
            fresh = _local(
                prompt
                + "\n\nYour line nearly repeats an earlier line's phrasing. Rewrite COMPLETELY - "
                "new words, no recycled filler, keep THIS line's facts and numbers.",
                temperature=0.3, tag="theme", repeat_penalty=1.3,
            ).strip()
            if fresh and _numbers_survived(s, fresh, fact_need):
                frag = re.split(r"(?<=[.!?])\s+", fresh.strip())
                if len(frag) > 2:
                    fresh = " ".join(frag[:2])
                line = fresh.strip()
        re_voiced.append(line)
    themed = "\n".join(re_voiced)
    got = _count_sentences(themed)
    want = len(sentences)
    if got != want:
        print(f"    [theme] {got} output sentences vs {want} original - tolerance fallback applied")
    return themed