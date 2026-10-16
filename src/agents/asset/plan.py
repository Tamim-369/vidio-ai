"""Shot planner: understand the story, decide what each beat should SHOW.

WHY THIS EXISTS
---------------
The asset stage used to search whatever words the query agent emitted. That
works when the topic owns a real, photographed subject ("USS Akagi"): Commons
has files whose titles match the name. It falls apart for a NARRATIVE topic.
A private missing-person story ("Resolved. Patrick O'Sullivan disappeared in
1992...") reached the fetcher as entities like "Resolved", "social media" and
"Ireland", and the event anchor glued the whole headline onto each one --
producing searches like "Ireland Resolved. Patrick O'Sullivan disappeared".
No archive holds that file. Every query was a guaranteed miss, and the fetch
loop then spent its refetch rounds re-asking the same dead questions.

The planner replaces "search the words" with "understand the shot". One local
call reads the TOPIC and the SCRIPT and answers, per beat:
  - what should this shot SHOW (a visual intent),
  - is it a real, findable subject (Commons archive) or a generic/symbolic one
    (representative stock imagery),
  - short, source-appropriate search queries.

It also classifies the whole topic into a mode and lists only the subjects that
genuinely have public archival photographs, so the entity pool stops being
built from private people the archive cannot answer.

Never raises: an unparseable reply returns None and the caller keeps its
existing entity + query behaviour (a bad plan must never be worse than no plan).
"""
from __future__ import annotations

import json
import os
import re
import threading

from src.agents.common.llm import _local

# Planning is one call but it must never become the pipeline's slowest step.
# On Groq it lands in ~2s; on a local-only box gemma3:4b generating a plan for a
# long script measured ~2 minutes. Cap it: past the budget we return None and
# the fetch falls back to the entity + query-agent path, which is what the
# pipeline did before the planner existed.
PLAN_TIMEOUT_S = float(os.getenv("ASSET_PLAN_TIMEOUT", "45"))

_MODES = {"archival", "modern", "symbolic"}
_KINDS = {"entity", "generic", "symbolic"}

_PLAN_PROMPT = """You are the shot planner for a documentary short-video channel. \
You read the VIDEO TOPIC and its NARRATION LINES and decide what each beat \
should SHOW.

The story may be about a famous, photographed subject (a warship, a battle, a \
public figure) OR about a private, ordinary person who has NO public \
photographs. Telling those apart is your first job.

STEP 1 - CHOOSE THE MODE:
- "archival": an event/era with real archive or press photos (a WWII battle, a \
1970s disaster, a famous public figure, a historic city).
- "modern": recent or contemporary and photographed, but not archival (a 2024 \
technology or business story).
- "symbolic": the main subject is a private person or an emotion with NO real \
photograph (a missing man, a family's grief, "silence", "regret", "34 years").

STEP 2 - LIST real_subjects:
- ONLY named subjects that genuinely have PUBLIC archival photographs: ships, \
cities, countries, monuments, famous people, named events.
- A private individual who merely appears in the story MUST NOT be listed. If \
in doubt, leave it out.
- If there are none, return an empty list.

STEP 3 - PLAN EVERY LINE (key = the line number):
- "kind":
  - "entity"  - the line is about one of your real_subjects.
  - "generic" - a real but unnamed subject a photo agency would tag (a man's \
face, a phone screen, an Irish street, a police car).
  - "symbolic"- an abstract or emotional beat with no literal photo (grief, \
time passing, a truth revealed).
- "intent": 3-8 words naming the SHOT - what a camera should point at.
- "queries": 1-3 short search strings.
  - For "entity": the exact name plus era, as an archive file search \
("MG42 machine gun 1940s", "USS Enterprise CV-6").
  - For "generic"/"symbolic": 2-5 plain descriptive words a stock library would \
tag ("man looking at phone", "Irish countryside", "empty chair by window"). \
NEVER a private person's name; NEVER the sentence's literal words.

RULES:
- NEVER invent names or facts that are not in the topic.
- A number, a feeling or a metaphor is not a picture; translate it to a real image.
- 4-6 words max per query. No quotes, no minus signs, no "photo of".

VIDEO TOPIC:
{topic}

NARRATION LINES (number them exactly):
{script}

Reply with ONLY this JSON, no markdown, no explanation:
{{"mode":"archival|modern|symbolic","year":<year or null>,
  "real_subjects":["..."],
  "shots":{{"1":{{"kind":"entity|generic|symbolic","intent":"...","queries":["..."]}}}}}}"""


def _coerce_queries(raw) -> list:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out, seen = [], set()
    for q in raw:
        s = re.sub(r"\s+", " ", str(q or "").strip().strip(".,;:!?\"'`")).strip()
        if not s or len(s) < 2 or len(s) > 70:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= 3:
            break
    return out


def _coerce_year(raw):
    try:
        y = int(raw)
    except (TypeError, ValueError):
        return None
    return y if 1400 <= y <= 2100 else None


def parse_plan(raw: str, line_ids: list | None = None) -> dict | None:
    """Parse a planner reply into a validated plan, or None if unusable."""
    if not raw or not raw.strip():
        return None
    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    data = None
    for candidate in (txt, *(m.group(0) for m in re.finditer(r"\{.*\}", txt, re.S))):
        try:
            data = json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict):
            break
        data = None
    if not isinstance(data, dict):
        return None

    mode = str(data.get("mode") or "").strip().lower()
    if mode not in _MODES:
        mode = "archival"

    real_subjects = []
    for s in (data.get("real_subjects") or []):
        s = re.sub(r"\s+", " ", str(s or "").strip())
        if s and s not in real_subjects:
            real_subjects.append(s)

    shots: dict = {}
    raw_shots = data.get("shots")
    if isinstance(raw_shots, dict):
        items = raw_shots.items()
    elif isinstance(raw_shots, list):
        items = []
        for entry in raw_shots:
            if isinstance(entry, dict) and entry.get("id") is not None:
                items.append((entry.get("id"), entry))
    else:
        items = []

    wanted = set(line_ids) if line_ids else None
    for key, shot in items:
        if not isinstance(shot, dict):
            continue
        try:
            lid = int(str(key).strip())
        except (TypeError, ValueError):
            continue
        if wanted is not None and lid not in wanted:
            continue
        kind = str(shot.get("kind") or "").strip().lower()
        if kind not in _KINDS:
            kind = "generic"
        queries = _coerce_queries(shot.get("queries"))
        intent = re.sub(r"\s+", " ", str(shot.get("intent") or "").strip())[:120]
        shots[lid] = {"kind": kind, "intent": intent, "queries": queries}

    if not shots:
        return None
    return {
        "mode": mode,
        "year": _coerce_year(data.get("year")),
        "real_subjects": real_subjects,
        "shots": shots,
    }


def plan_shots(topic: str, lines: list, model: str | None = None) -> dict | None:
    """One local call: topic + script -> shot plan (or None on failure).

    Never raises. A None result means "no plan"; callers fall back to the
    existing entity + query-agent behaviour.
    """
    if not lines:
        return None
    script = "\n".join(
        f'{ln["id"]}. {(ln.get("text") or "").strip()}'
        for ln in lines
        if str(ln.get("text") or "").strip()
    )
    if not script:
        return None
    box: dict = {}

    def _run():
        try:
            box["raw"] = _local(
                _PLAN_PROMPT.format(topic=(topic or "unknown"), script=script),
                temperature=0.2,
                tag="plan",
                model=model,
            )
        except Exception as e:  # noqa: BLE001
            box["err"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(PLAN_TIMEOUT_S)
    if "raw" not in box and "err" not in box:
        print(f"  [plan] planner timed out after {PLAN_TIMEOUT_S:.0f}s - using entity/query fallback")
        return None
    if "err" in box:
        print(f"  [plan] planner unavailable ({type(box['err']).__name__}) - using entity/query fallback")
        return None
    plan = parse_plan(box["raw"], line_ids=[ln.get("id") for ln in lines])
    if plan is None:
        print("  [plan] planner returned no usable plan - using entity/query fallback")
    return plan
