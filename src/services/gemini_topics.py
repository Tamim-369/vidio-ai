"""Gemini-powered viral topic brainstormer.

One grounded Gemini call produces a batch of fresh, on-niche topic seeds.
Grounding (Google Search) anchors each idea in real, documented history so
we don't ship hallucinated subjects, and the blocklist is fed into the prompt
plus enforced again in code — uniqueness is guaranteed twice.

Requires GEMINI_API_KEY in .env. Falls back to None (caller decides) if the
key is missing or the API is unreachable.
"""
import json
import os
import re
import time

from google import genai
from google.genai import types

from src.config.settings import GEMINI_API_KEY, GEMINI_MODEL

PROMPT = """You brainstorm viral topic ideas for a faceless YouTube Shorts channel.

NICHE: documented dark history / secret military experiments / engineering
disasters / forgotten events. The 3 money angles are: MYSTERY (secrets,
cover-ups, unsolved unknowns), SCARY (horror and terror in the story itself),
INJUSTICE (institutional betrayal, innocent victims, cruel neglect). A 4th
acceptable angle is HEROIC (underdog soldiers, impossible stands).

RULES:
- Each topic must be a CONCRETE named subject: a specific person, operation,
  place, event, or experiment. No vague concepts like "secrets of the CIA".
- It must be REAL and DOCUMENTED. Use your search grounding to verify each one
  actually happened and has a factual record. Never invent or embellish.
- NEVER repeat any title in the BLOCKLIST below (exact or near-identical).
- Prefer topics that are little-known but well-documented — a dramatic story
  most people have never heard.
- No true crime, no serial killers, no missing-person cases, no paranormal/
  cryptid/folklore (ghosts, aliens, skinwalkers, giants, monsters), no boring
  weapon/aircraft spec catalogs.

Return ONLY a JSON array of {target} objects, each:
{{"title": "short topic title (max 12 words)", "angle": "mystery|scary|injustice|heroic", "summary": "1-2 sentence hook why this is fascinating"}}

BLOCKLIST (already used or too close to used topics — avoid all):
{blocklist}
"""

MAX_TOPICS_PER_CALL = 20


def brainstorm_topics(count: int = 20, blocklist: list = None) -> list:
    """Return a list of {title, angle, summary} topic seeds from Gemini.

    Returns an empty list if the API key is missing or the call fails —
    the caller falls back to the other topic sources.
    """
    if not GEMINI_API_KEY:
        print("  [gemini] No GEMINI_API_KEY — skipped")
        return []

    blocked = blocklist or []
    prompt = PROMPT.format(
        target=min(count, MAX_TOPICS_PER_CALL),
        blocklist="\n".join(f"- {b}" for b in blocked) if blocked else "(none)",
    )

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = None
    # Grounding (Google Search) is nice-to-have but draws from a separate
    # quota that's often exhausted on the free tier — retry plain if it 429s.
    for tools in (
        [types.Tool(google_search=types.GoogleSearch())],
        None,
    ):
        try:
            chat = client.chats.create(
                model=GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.9,  # diverse angles; dedup handled in code
                    tools=tools,
                ),
            )
            response = chat.send_message(prompt)
            break
        except Exception as e:
            if tools is not None:
                print(f"  [gemini] grounding unavailable ({e}), retrying without...")
                continue
            print(f"  [gemini] API error: {e}")
            return []

    text = response.text or ""
    try:
        m = re.search(r"\[.*\]", text, re.S)
        data = json.loads(m.group(0))
    except Exception:
        print("  [gemini] Could not parse response")
        return []

    seeds = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        angle = str(item.get("angle", "mystery")).strip()
        if angle not in ("mystery", "scary", "injustice", "heroic"):
            angle = "mystery"
        seeds.append({
            "title": title,
            "angle": angle,
            "summary": str(item.get("summary", "")).strip(),
        })
    return seeds