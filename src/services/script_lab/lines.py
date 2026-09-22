"""Stage 4 - LINES: themed prose -> structured script lines (beat + tone + text).

Each line becomes the shortest unit the rest of the video pipeline needs to
render: an image, a TTS clip, and a caption. One re-ask exists for truncated
output; anything the model emits is normalized through `_parse_lines`.
"""
from __future__ import annotations

from src.config.script_lab_prompts import SCRIPT_JSON_PROMPT
from src.services.script_lab.llm import _local
from src.services.script_lab.text import _count_sentences, _loads_json, _render


def to_lines(prose: str) -> list[dict]:
    prompt = _render(SCRIPT_JSON_PROMPT, prose=prose)
    out = _local(prompt, temperature=0.2, tag="json")
    lines = _parse_lines(out)
    nprose = _count_sentences(prose)
    if len(lines) < max(2, nprose):  # likely truncated
        print(f"    [json] only {len(lines)} lines (expect ~{nprose}) - re-asking once")
        out2 = _local(
            prompt
            + "\n\nYou stopped early. Convert the ENTIRE script into 1 JSON object per line, "
            "in order, one line of the script per object. Do not truncate.",
            temperature=0.2, tag="json",
        )
        lines = _parse_lines(out2)
    merged = []
    i = 0
    while i < len(lines):
        ln = dict(lines[i])
        while ln["text"].endswith(":") and i + 1 < len(lines):  # label dropped its sentence
            i += 1
            ln["text"] = ln["text"] + " " + (lines[i].get("text") or "").strip()
            ln["beat"] = lines[i].get("beat", ln.get("beat"))
        merged.append(ln)
        i += 1
    return merged


def _parse_lines(out: str) -> list[dict]:
    lines = _loads_json(out)
    if isinstance(lines, dict):
        lines = next((v for v in lines.values() if isinstance(v, list)), None)
    if not isinstance(lines, list) or not lines:
        raise ValueError(f"json stage: expected a non-empty list of lines: {out[:160]!r}")
    if isinstance(lines[0], list):  # salvage nested: [[...], ...]
        lines = lines[0]
    out_list = []
    for ln in lines:
        if isinstance(ln, dict) and ln.get("text"):
            ln = dict(ln)
            ln["text"] = (ln.get("text") or "").strip()
            ln.setdefault("beat", "setup")
            ln.setdefault("tone", "normal")
            out_list.append(ln)
    if not out_list:
        raise ValueError(f"json stage: no usable line objects: {out[:160]!r}")
    return out_list