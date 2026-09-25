"""Scene agent prompts: pick ONE gripping incident from raw material.

The SCENE pass is the story's foundation: it reads the material and returns a
JSON brief (incident/person/stakes/detail/outcome/who_else/numbers). The writer
is deliberately given ONLY this brief, never the full article, so it cannot
retell the source in order — it must dramatize the single scene it is handed.
"""
# ---------------------------------------------------------------------- prompt

SCENE_SELECT_PROMPT = """You are a short-video story editor. Read the material below and pick the ONE most gripping incident in it - the single moment with the most at stake, the most dramatic, the most worth telling.

Rules
- Pick ONE incident, not a biography, not a trend, not the whole war. A single moment or tight sequence a viewer can picture.
- Prefer an incident that has: a clear person (or small group), real stakes, a visual, a turn. Avoid broad overview or chronological milestones.
- Only use facts the material actually states - never invent names, numbers, times, or places.
- The OUTCOME is required: you must say how this incident actually ended, exactly as the material says (failed, broke through, was ordered to dissolve, repelled the attack, etc.). A story built from an incident without its true ending will invent one.

Output ONLY a JSON object with exactly these keys:
{"incident": "what happened, in 1-2 sentences, using only the material's facts - INCLUDING how it ended", "person": "the one person or small group it centers on", "stakes": "what was on the line - the exact stakes stated in the material", "detail": "one concrete, vivid, specific detail from the material that makes this scene unforgettable", "outcome": "how the incident ended, verbatim facts only (e.g. 'the attack failed and the division was ordered to dissolve')", "who_else": ["every other named person, unit, division, group, or place involved in THIS scene", "list them all, exactly as written in the material"], "numbers": ["every exact number, date, and figure that matters in THIS scene", "as raw values, e.g. \"June 17\", \"18\", \"1940\", \"fifty men\""]}

MATERIAL:
{story}

Return ONLY the JSON."""