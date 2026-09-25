"""Story agent: an incident brief -> a short, dramatized, fact-gated story.

Two model-touching personalities plus a deterministic backend:
  SCENE  (delegated to src.agents.scene) picks the one incident to dramatize.
  WRITER (this module) drafts / revises the incident in the speaking style.
  SHOWRUNNER (this module) verdicts drafts: PASS airs, REVISE rewrites with
  line-level notes, REJECT abandons the scene for a new pick.

Deterministic shaping and the fact gate live in :mod:`helpers` (no model).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from src.agents.common.llm import _local
from src.agents.common.text import _loads_json, _render, _scalar
from src.agents.scene.agent import pick_scene
from src.agents.story.helpers import (
    _fact_gate,
    _fact_notes,
    _pick_best,
    _postprocess,
)
from src.agents.story.prompts import (
    EVALUATOR_PROMPT,
    REVISE_PROMPT,
    STORY_PROMPT,
)
from src.agents.story.styles import get_speaking_style

# Story writer <-> showrunner back-and-forth budget. The evaluator decides
# where the writer goes each round: PASS (accept immediately), REVISE
# (rewrite with its line-level notes, then re-judge), REJECT (this scene
# cannot become a story -> pick a new scene). The evaluator is biased to PASS,
# so a good draft exits after the FIRST judge call and the loop only spends
# time when a draft is genuinely weak. MAX_EVAL_ROUNDS = how many rewrite
# attempts a weak draft may get; MAX_SCENE_ATTEMPTS caps scene re-picks before
# the topic is skipped. Deep-polish mode can raise MAX_EVAL_ROUNDS=3 at the
# cost of minutes per story.
MAX_EVAL_ROUNDS = int(os.getenv("MAX_EVAL_ROUNDS", "1"))
MAX_SCENE_ATTEMPTS = int(os.getenv("MAX_SCENE_ATTEMPTS", "2"))
# Deterministic fact-gate: runs in microseconds BEFORE any showrunner vote, so
# a good draft that reuses only the scene brief's facts is never slowed. A
# draft that invents a number, a magnitude, or a victory-out-of-defeat is sent
# back to the writer with the exact fact - each fix costs one story call. If a
# draft still violates the gate after MAX_FACT_FIXES fixes, the scene is
# rejected (the batch moves to the next topic) rather than shipping a lie.
MAX_FACT_FIXES = int(os.getenv("MAX_FACT_FIXES", "1"))


def _render_story(brief: dict, block: str) -> str:
    return _render(
        STORY_PROMPT,
        style=block,
        incident=brief.get("incident") or "",
        person=brief.get("person") or "",
        stakes=brief.get("stakes") or "",
        detail=brief.get("detail") or "",
        outcome=brief.get("outcome") or "",
        who_else=brief.get("who_else") or "",
        numbers=brief.get("numbers") or "",
    )


def _evaluate(draft: str, brief: dict) -> tuple[str, str]:
    """Pass 3: the showrunner verdicts the draft.

    Returns (verdict, notes). verdict is one of PASS/REVISE/REJECT. A verdict
    that cannot be parsed is treated as REVISE so the writer gets a chance to
    fix the draft rather than silently accepting junk.
    """
    out = _local(
        _render(
            EVALUATOR_PROMPT,
            draft=draft,
            incident=brief.get("incident") or "",
            person=brief.get("person") or "",
            stakes=brief.get("stakes") or "",
            detail=brief.get("detail") or "",
        ),
        temperature=0.2,
        tag="eval",
    )
    verdict = "REVISE"
    notes = (out or "").strip()
    try:
        obj = _loads_json(out or "")
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        v = _scalar(obj.get("verdict"))
        if v:
            v = v.upper()
            if "PASS" in v:
                verdict = "PASS"
            elif "REJECT" in v:
                verdict = "REJECT"
            elif "REVISE" in v:
                verdict = "REVISE"
        notes = _scalar(obj.get("notes")) or notes
    return verdict, notes


def _refine(previous: str, notes: str, brief: dict, block: str) -> str:
    """Pass 2b: the writer rewrites its own draft under the showrunner's notes."""
    return _local(
        _render(
            REVISE_PROMPT,
            style=block,
            incident=brief.get("incident") or "",
            person=brief.get("person") or "",
            stakes=brief.get("stakes") or "",
            detail=brief.get("detail") or "",
            outcome=brief.get("outcome") or "",
            who_else=brief.get("who_else") or "",
            numbers=brief.get("numbers") or "",
            previous=previous,
            notes=notes,
        ),
        temperature=0.6,
        tag="story",
    )


def _gate_and_fix(draft: str, brief: dict, block: str, topic: str) -> str | None:
    """Deterministically clean a draft of invented facts.

    Runs the fact-gate (free) and, when it finds a violation, sends the writer
    back with the EXACT fact via `_refine`, up to MAX_FACT_FIXES. Returns the
    cleaned draft, or None when the writer is still inventing facts after the
    budget - that scene must not ship, so the caller treats it like a REJECT.
    """
    for _ in range(MAX_FACT_FIXES):
        problems = _fact_gate(draft, brief)
        if not problems:
            return draft
        print(f"     …fact-gate: {len(problems)} violation(s)")
        for p in problems:
            print(f"        - {p}")
        try:
            raw = _refine(draft, _fact_notes(problems), brief, block)
            draft = _postprocess(raw or "", block, topic)
        except (ValueError, RuntimeError) as e:
            print(f"     …fact-fix unusable ({e}) — accepting draft as-is.")
            problems = _fact_gate(draft, brief)
            return draft if not problems else None
    return draft if not _fact_gate(draft, brief) else None


def _write_draft(brief: dict, block: str, topic: str) -> str | None:
    """Pass 1: writer produces a post-processed draft for `brief`.

    Returns None when the raw STORY reply is unusable (wedged/empty reply) so
    the caller treats this like a rejected scene attempt.
    """
    try:
        raw = _local(_render_story(brief, block), temperature=0.6, tag="story")
        text = (raw or "").strip()
        return _postprocess(text, block, topic)
    except (ValueError, RuntimeError) as e:
        print(f"     …draft unusable ({e}) — treating as rejected scene.")
        return None


def _try_rewrite(draft: str, notes: str, brief: dict, block: str, topic: str) -> str | None:
    """Rewrite under showrunner notes, re-running the deterministic fact-gate.

    Returns the refined fact-clean draft, or None when the rewrite is unusable
    or invents facts it cannot be fixed from (caller keeps the best draft).
    """
    try:
        raw = _refine(draft, notes, brief, block)
        refined = _postprocess(raw or "", block, topic)
    except (ValueError, RuntimeError) as e:
        print(f"     …refinement unusable ({e}) - accepting best draft.")
        return None
    # The rewrite is a NEW draft - re-run the deterministic fact-gate on
    # it. A rewrite that smuggles in invented facts must not air either.
    refined = _gate_and_fix(refined, brief, block, topic)
    if refined is None:
        print(
            "     …rewrite invented facts and could not be fixed - "
            "accepting best draft instead."
        )
        return None
    return refined


def _showrunner_loop(draft: str, brief: dict, block: str, topic: str) -> str | None:
    """Vote/re-write loop on one scene. Returns the shipped story or None.

    None means the showrunner REJECTed the scene (caller must pick a new one);
    otherwise returns the greenlit (PASS or best-at-budget) story string.
    """
    best = draft
    for rnd in range(MAX_EVAL_ROUNDS + 1):
        print(f"  [showrunner pass {rnd + 1}]")
        verdict, notes = _evaluate(draft, brief)
        print(f"     verdict: {verdict}")
        if verdict == "PASS":
            print("     …greenlit as-is.")
            return _pick_best(best, draft)
        if verdict == "REJECT":
            print("     …this scene cannot become a story - moving on.")
            return None
        if notes:
            print(f"     notes:   {notes[:200]}")
        if rnd == MAX_EVAL_ROUNDS:
            print(f"     …still weak after {MAX_EVAL_ROUNDS} rewrite(s), accepting best draft.")
            return best
        refined = _try_rewrite(draft, notes, brief, block, topic)
        if refined is None:
            return best
        best = _pick_best(best, refined)
        draft = refined
    return best


def build_story(story: str, style: str = "narrator", topic: str = "", scene: dict | None = None) -> str:
    """Turn `story` into a short, detailed, easy-to-follow narrative.

    Writer/showrunner back-and-forth: a draft is written, THE SHOWRUNNER (a
    separate ruthless personality, biased to PASS) verdicts it. A strong draft
    is greenlit after the single first vote; a REVISE triggers one rewrite with
    the editor's notes, then a fresh vote. Loops at most MAX_EVAL_ROUNDS
    rewrites (deep-polish mode raises it; default speed mode: one). REJECT
    abandons the scene, picks a new one, and starts the loop again - up to
    MAX_SCENE_ATTEMPTS. All scenes rejected raises ValueError so the batch
    runner moves to the next topic instead of shipping a dead script.

    `style` selects the speaking style injected into the prompt (narrator,
    arnold, trump, andrew_tate). `topic` (the topic title) is used to strip a
    leading title echo that small models copy from the article's H1. `scene`
    is an already-picked incident brief (from the pipeline driver); when None,
    the scene pass runs here. Returns the story as a string (one sentence per
    line). Raises on empty model output so a wedged reply cannot flow on.
    """
    block = get_speaking_style(style)
    brief = scene if isinstance(scene, dict) else pick_scene(story)
    rejected = []
    for attempt in range(MAX_SCENE_ATTEMPTS):
        if attempt > 0:
            brief = pick_scene(story, avoid=rejected)
        print(f"\n[scene attempt {attempt + 1}/{MAX_SCENE_ATTEMPTS}]")
        print(f"     incident: {brief.get('incident', '')[:120]}")

        draft = _write_draft(brief, block, topic)
        if draft is None:
            # A wedged STORY reply is a failed scene attempt, not a
            # batch-killer: treat it like the showrunner rejecting this scene.
            rejected.append(brief.get("incident") or "")
            continue

        # DETERMINISTIC FACT-GATE: free (microseconds) and runs BEFORE any
        # showrunner vote. A clean draft - using only the brief's numbers and
        # the brief's true outcome - sails straight to the showrunner. A draft
        # that invented a number, a magic magnitude, or a victory out of defeat
        # is fixed here with the exact fact. The showrunner only ever votes on
        # a factually-consistent draft. If a fact-clean version cannot be
        # produced within MAX_FACT_FIXES, the scene is rejected like a REJECT
        # vote: the batch moves to the next topic.
        draft = _gate_and_fix(draft, brief, block, topic)
        if draft is None:
            print(
                "     …still inventing facts after the fix budget - "
                "rejecting scene, never shipping a lie."
            )
            rejected.append(brief.get("incident") or "")
            continue

        shipped = _showrunner_loop(draft, brief, block, topic)
        if shipped is not None:
            return shipped
        rejected.append(brief.get("incident") or "")

    raise ValueError(
        "story stage rejected every candidate scene - cannot build a story from this material"
    )