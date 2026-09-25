"""LLM prompts for the research agent."""

IDEAS_PROMPT = """You are a YouTube content strategist for a faceless channel about {niche}.

Given the raw source material below, extract up to 3 video ideas.

For each idea output ONLY valid JSON in this schema:
{{
  "title": "clickable but not clickbait-lie title, under 70 chars",
  "hook": "first 15 seconds script, must create an open question",
  "why_viral": "one sentence on the curiosity gap or stakes",
  "outline": ["beat 1", "beat 2", "beat 3", "beat 4"],
  "source_url": "...",
  "confidence": 1-10 (how obscure/novel is this, avoid oversaturated topics)
}}

Reject anything that is: already extremely well-covered on YouTube
(D-Day, Roswell, Atlantis basics, Titanic), unverifiable pure speculation
with zero primary source, or requires reproducible technical/harmful detail.
Prefer stories a general audience has NOT heard — obscure but documented.

Return ONLY a JSON array of idea objects, no prose, no markdown fences.

Raw material:
{raw}
"""

LIGHT_IDEAS_PROMPT = """List documentary video titles about {subject}.
Titles must be under 70 characters, use strong action verbs, and name specific, real, confirmed events, units, or figures from documented history (legends and supernatural folklore are allowed when framed as a story or legend — no true-crime serial killers or missing-person cases). Prefer stories a general audience has not already seen everywhere.
Return ONLY a JSON array of strings, e.g. ["Title one", "Title two"], and nothing else.
"""