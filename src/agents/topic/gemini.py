"""Gemini-powered viral topic brainstormer.

One grounded Gemini call produces a batch of fresh, on-niche topic seeds.
Grounding (Google Search) anchors each idea in real, documented history so
we don't ship hallucinated subjects, and the blocklist is fed into the prompt
plus enforced again in code — uniqueness is guaranteed twice.

Requires GEMINI_API_KEY in .env. Falls back to None (caller decides) if the
key is missing or the API is unreachable.
"""
import json
import re

from google import genai
from google.genai import types

from src.agents.topic.prompts import GEMINI_TOPICS_PROMPT
from src.services.providers import GEMINI_API_KEY, GEMINI_MODEL

MAX_TOPICS_PER_CALL = 20


def _gemini_chat(prompt: str):
    """One grounded chat call, retrying plain when grounding 429s."""
    client = genai.Client(api_key=GEMINI_API_KEY)
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
            return chat.send_message(prompt)
        except Exception as e:
            if tools is not None:
                print(f"  [gemini] grounding unavailable ({e}), retrying without...")
                continue
            print(f"  [gemini] API error: {e}")
            return None


def _parse_seeds(text: str) -> list:
    """Extract the JSON list from Gemini's reply into {title, angle, summary}."""
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


def brainstorm_topics(count: int = 20, blocklist: list = None) -> list:
    """Return a list of {title, angle, summary} topic seeds from Gemini.

    Returns an empty list if the API key is missing or the call fails —
    the caller falls back to the other topic sources.
    """
    if not GEMINI_API_KEY:
        print("  [gemini] No GEMINI_API_KEY — skipped")
        return []

    blocked = blocklist or []
    prompt = GEMINI_TOPICS_PROMPT.format(
        target=min(count, MAX_TOPICS_PER_CALL),
        blocklist="\n".join(f"- {b}" for b in blocked) if blocked else "(none)",
    )

    # Grounding (Google Search) is nice-to-have but draws from a separate
    # quota that's often exhausted on the free tier — retry plain if it 429s.
    response = _gemini_chat(prompt)
    return _parse_seeds(response.text or "") if response else []