"""Script agent: themed prose -> structured JSON script lines.

The single agent call produces the whole final script: each line carries
text + beat + tone. Image queries are NOT written here any more - they need
the topic as ground truth, so a DEDICATED query-builder stage owns them.
Anything the model emits is normalized through the helpers, and a truncated
reply gets one re-ask.
"""
from __future__ import annotations

from src.agents.common.llm import _local
from src.agents.common.text import _count_sentences, _render
from src.agents.script.helpers import _fold_fragments, _parse_lines, _postprocess, _text_of
from src.agents.script.prompts import SCRIPT_AGENT_PROMPT


def to_lines(prose: str) -> list[dict]:
    """Turn the themed narration into the final script JSON (one agent call)."""
    prompt = _render(SCRIPT_AGENT_PROMPT, prose=prose)
    out = _local(prompt, temperature=0.2, tag="json")
    lines = _parse_lines(out)
    nprose = _count_sentences(prose)
    # Only re-ask when clearly truncated (well under the expected count).
    # A one-line shortfall (11 vs 12) is normal - fragments get merged - and
    # forcing a retry risks trading a good reply for a cut-off one.
    if len(lines) < max(2, int(nprose * 0.7)):
        print(f"    [agent] only {len(lines)} lines (expect ~{nprose}) - re-asking once")
        out2 = _local(
            prompt
            + "\n\nYou stopped early. Convert the ENTIRE script into 1 JSON object per line, "
            "in order, one line of the script per object. Do not truncate.",
            temperature=0.2, tag="json",
        )
        lines2 = _parse_lines(out2)
        if len(lines2) > len(lines):
            lines = lines2
    merged = []
    i = 0
    while i < len(lines):
        ln = dict(lines[i])
        while ln["text"].endswith(":") and i + 1 < len(lines):  # label dropped its sentence
            i += 1
            tail = _text_of(lines[i].get("text"))
            ln["text"] = ln["text"] + " " + tail if tail else ln["text"]
            ln["beat"] = _text_of(lines[i].get("beat")) or ln.get("beat")
        merged.append(ln)
        i += 1
    merged = _postprocess(merged)
    merged = _fold_fragments(merged)
    return merged