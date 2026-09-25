"""Script agent - deterministic helpers (no model calls).

Normalization, capping, and shape fixes for the JSON lines the agent emits.
These are never trusted to the model: the hook must be punchy (never a drop),
the closer can't be dead air, echo lines get dropped, and fragments fold back
into the line they confirm.
"""
from __future__ import annotations

import re

from src.agents.common.text import _loads_json

_MAX_LINES = 12  # hard cap: short-format video, one line per sentence

# Closer lines that add nothing but dead air - never allowed as the last line.
_CLOSER_FILLER = re.compile(r"^\s*(the end|that's it|that is it|thanks? for watching"
                            r"|thank you for watching|subscribe( for more)?)\s*\.?\s*$",
                            re.I)


def _text_of(v) -> str:
    """Flatten any model-provided field to a trimmed string (never crashes)."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return " ".join(p for p in (_text_of(x) for x in v) if p).strip()
    if isinstance(v, dict):
        return _text_of(next((v[k] for k in ("text", "value", "beat") if k in v), ""))
    return str(v).strip()


def _parse_lines(out: str) -> list[dict]:
    lines = _loads_json(out)
    if isinstance(lines, dict):
        lines = next((v for v in lines.values() if isinstance(v, list)), None)
    if not isinstance(lines, list) or not lines:
        raise ValueError(f"agent stage: expected a non-empty list of lines: {out[:160]!r}")
    if isinstance(lines[0], list):  # salvage nested: [[...], ...]
        lines = lines[0]
    out_list = []
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        text = _text_of(ln.get("text"))
        if not text:
            continue
        ln = dict(ln)
        ln["text"] = text
        ln["beat"] = _text_of(ln.get("beat")) or "setup"
        ln["tone"] = _text_of(ln.get("tone")) or "normal"
        out_list.append(ln)
    if not out_list:
        raise ValueError(f"agent stage: no usable line objects: {out[:160]!r}")
    return out_list


def _fold_fragments(lines: list[dict]) -> list[dict]:
    """Fold a line with almost no words into the previous line.

    The story stage merges short catchphrase tags, but the AGENT often splits
    them back out as their own JSON lines ("It's true.", "A complete one.") -
    as videos those are 1-2 seconds of dead air with bad image queries.
    """
    if not lines:
        return lines
    out = [lines[0]]
    for ln in lines[1:]:
        if len(re.findall(r"[a-z0-9']+", _text_of(ln.get("text")).lower())) < 4:
            out[-1] = dict(out[-1])
            prev_text = _text_of(out[-1].get("text"))
            out[-1]["text"] = f"{prev_text} {_text_of(ln.get('text'))}".strip()
        else:
            out.append(ln)
    return out


def _run_len(a: list[str], b: list[str]) -> int:
    """Longest contiguous token run shared by two lists (simple grid DP)."""
    prev = [0] * (len(b) + 1)
    best = 0
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _drop_echo_lines(lines: list[dict]) -> list[dict]:
    """Drop a line that re-says a full clause from an earlier line.

    The AGENT duplicates too - e.g. the laureate closer restating the payoff
    sentence it already gave five lines earlier. The story-stage dedup cannot
    see this (it happens after), so the same run-guard runs here.
    """
    if len(lines) < 2:
        return lines
    kept: list[dict] = []
    seen: list[list[str]] = []
    for ln in lines:
        toks = re.findall(r"[a-z0-9']+", _text_of(ln.get("text")).lower())
        if not toks:
            kept.append(ln)
            continue
        if any(_run_len(toks, prev) >= 8 for prev in seen):
            continue
        kept.append(ln)
        seen.append(toks)
    return kept or lines


def _postprocess(lines: list[dict]) -> list[dict]:
    """Deterministic script-shape fixes (never trusted to the model):
    - hard cap at _MAX_LINES
    - the hook must be punchy, never a drop
    - the closer can't be "The end."-style dead air
    """
    if len(lines) <= _MAX_LINES:
        lines = list(lines)
    else:
        # keep the hook (first), the payoff/peak, and the closer; drop bloat
        # from the middle so the tight 8-12 arc survives a 40-line blow-up.
        keep = [lines[0]]
        body = lines[1:-1]
        # prefer keeping any payoff/escalate beats, then fill evenly.
        for ln in body:
            if ln.get("beat") in ("payoff", "escalate") and len(keep) < _MAX_LINES - 1:
                keep.append(ln)
        for ln in body:
            if len(keep) >= _MAX_LINES - 1:
                break
            if ln not in keep:
                keep.append(ln)
        keep.append(lines[-1])
        lines = keep[:_MAX_LINES]
    while len(lines) > 1 and _CLOSER_FILLER.match(_text_of(lines[-1].get("text"))):
        lines.pop()
    for ln in lines:
        if _text_of(ln.get("beat")).lower() == "hook" and _text_of(ln.get("tone")).lower() == "drop":
            ln["tone"] = "normal"
    lines = _drop_echo_lines(lines)
    return lines