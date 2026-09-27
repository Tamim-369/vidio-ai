"""YouTube metadata prompts.

Only metadata generation lives here now. get_raw_script_prompt() wrote the
numbered "one sentence per line" script that quote_agent.py's whole design
centres on, and it went with the retired script_lab pipeline; the surviving
prompts only describe a finished script that is already written for the card.

These describe a parody-quote Shorts channel, where the on-screen "speaker" is
always a recognisable public figure (Trump, Arnold, Tate) and the line is a
fictional quote in their voice. The emotion to reach for is recognition and
amusement -- "that is exactly what he would say" -- not the dread and injustice
of a true-crime channel, which is what these prompts used to ask for.
"""

_MOOD = """The line is a parody quote written in a public figure's voice. The joke is
recognition: the viewer should think "that is exactly what he would say", then
laugh because it is also true. Reach for amusement, recognition, indignation or
surprise -- never for dread, grief or moral outrage, because nothing in this
video is actually dark and promising darkness is a lie the video cannot pay off."""

# This is the fallback for a script with no quote to build a title from, so it
# has to invent a short descriptive one. A planned batch never reaches it: those
# titles are the first line's own quote, which runs 87-104 characters, because a
# truncated quote is a missing punchline and that is the whole point of the
# video. The limit YouTube enforces is 100 characters; see TITLE_MAX_CHARS.
_TITLE_RULES = """===== TITLE RULES (a planned batch brings its own title; this is the fallback) =====
- 40-60 characters. A planned title is longer because it carries the whole
  quote, and cutting that quote is a worse trade than a wide title.
- The character's name inside the first 20 characters. A title that truncates
  before the name is a title nobody can attribute.
- At least ONE concrete specific: a number, a place, a body part, or an object.
  "Tate on discipline" is weaker than "Tate: legs over motivation".
- Amusement, recognition or indignation. Curiosity gaps and dread belong to a
  true-crime channel and read as bait here.
- Do NOT resolve the joke in the title, and do NOT promise anything the line
  does not deliver.
- Names stay capitalised. Trump, Arnold, Tate and any real name are proper nouns;
  do not lowercase or strip them to satisfy a "one capitalised word" rule.
- No emoji, no clickbait punctuation, no "YOU WONT BELIEVE"."""

_DESC_RULES = """===== DESCRIPTION RULES =====
- 40-80 words total. This is a 45-second Short, not an essay; a long description
  is padding the viewer never reads.
- First 100 characters are the ad copy, and they should name the character and
  the theme so search and the feed both have something to match.
- Say the video is a parody, plainly. Do not present a fabricated quote as real.
- Name the character once, naturally, in the first sentence.
- End with 2-3 hashtags, lowercase.
- Never invent quotes, dates, or claims the video does not make."""

_TAGS_RULES = """===== TAGS =====
- 8-15 tags, lowercase, no punctuation except hyphens. The character name first
  (so "donald trump" leads), then the theme, then the format."""


def get_metadata_prompt(topic, script_text):
    return f"""You are a YouTube SEO expert for a faceless parody-quote Shorts channel. Generate a title, description, and tags for a video.

{_MOOD}

Topic: {topic}

Video script (narration):
{script_text}

Return ONLY valid JSON (no markdown, no code fences) in this exact shape:
{{
  "title": "<40-60 char title>",
  "title_alternates": ["<alt title 1>", "<alt title 2>", "<alt title 3>"],
  "description": "<40-80 word description>",
  "tags": ["<tag1>", "<tag2>", "<tag3>", "..."]
}}

===== THINK FIRST (do this internally before writing the final JSON) =====

STEP 1 — Identify the character and the theme.
Which public figure is speaking, and what are they talking about? Name both.
That pair is the whole video.

STEP 2 — Draft 3 candidate titles using different structures:
1. Name + contrarian claim: "Arnold: <thing nobody says out loud>"
2. Name + concrete detail: "Tate: <the specific, absurd, physical thing>"
3. Clash or admission: "Trump admits <the thing>"

STEP 3 — Score them against the rules below. Keep the best.

{_TITLE_RULES}

{_DESC_RULES}

{_TAGS_RULES}"""


def get_metadata_verifier_prompt(topic, script_text, metadata_json):
    return f"""You are a ruthless YouTube metadata quality auditor for a faceless parody-quote Shorts channel. Your ONLY job is to verify the generated metadata meets every rule, and reject anything that would get scrolled past.

{_MOOD}

Topic: {topic}

Video script (narration):
{script_text}

Generated metadata to verify (JSON):
{metadata_json}

===== YOUR JOB =====
1. Read the script. Identify the character and the theme.
2. Audit the title against EVERY rule below. Be harsh. A title that is merely
   "fine" FAILS.
3. Audit the description and tags too.
4. Check the video does not promise something the script does not deliver.
5. Return ONLY valid JSON (no markdown, no code fences) in this exact shape:
{{
  "verdict": "PASS" or "FAIL",
  "title_score": <0-100>,
  "description_score": <0-100>,
  "emotion_triggered": "<amusement|recognition|indignation|surprise|none>",
  "fails": ["<exact rule that failed>", ...],
  "fixed_title": "<a better title that passes every rule, ONLY if verdict is FAIL, else \\"\\">",
  "fixed_description": "<a better description if needed, else \\"\\">",
  "fixed_tags": ["<tags>"]
}}

{_TITLE_RULES}

{_DESC_RULES}

{_TAGS_RULES}

If verdict is FAIL, the "fixed_title"/"fixed_description"/"fixed_tags" MUST be genuinely improved versions that pass every rule -- this is what gets used. Never return empty fixes on a FAIL."""


def get_metadata_fix_prompt(topic, script_text, old_metadata_json, verifier_json):
    return f"""You are a YouTube SEO expert for a faceless parody-quote Shorts channel. Your previous metadata was rejected by a verifier. Rewrite it to fully pass every rule.

{_MOOD}

Topic: {topic}

Video script:
{script_text}

Previous (rejected) metadata:
{old_metadata_json}

Verifier feedback:
{verifier_json}

Fix EVERY failing rule the verifier flagged. The title should land recognition or
amusement, and must name the character early. Output the corrected metadata as
valid JSON in this exact shape:
{{
  "title": "<40-60 char title>",
  "title_alternates": ["<alt title 1>", "<alt title 2>", "<alt title 3>"],
  "description": "<40-80 word description>",
  "tags": ["<tag1>", "<tag2>", "<tag3>", "..."]
}}
No markdown, no code fences."""
