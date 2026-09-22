"""Prompt set for the staged script-production pipeline (script_lab).

Stage order:
  1. FACTS  - extract concrete facts from a story (JSON, decision work only)
  2. STORY  - turn the facts into a gripping short spoken script (plain prose, NO JSON)
  3. THEME  - re-skin that exact story in the Arnold or Trump voice (prose -> prose)
  4. JSON   - convert themed prose into structured lines (JSON)
  5. QUERY  - per-line image search queries (>=2 per line, reuse with a gap)

The interestingness of the script comes from the STORY-layer instructions and
the theme-layer persona prompts (my design), NOT from trusting a big model.
Each stage asks the small local model to do one narrow, well-bounded job.
"""

# ---------------------------------------------------------------- 1. facts

FACTS_PROMPT = """Read the war story below and extract every concrete, citable fact into JSON.

A fact must contain something a viewer can point at: a number, a name, a place, a date, an equipment item, a unit, or a stated action. Ignore opinion and color. Quote facts exactly as stated - never invent, round, or guess a number that is not in the text.

Return ONLY a JSON array. No markdown, no prose.
[
  {"fact": "<short factual clause>", "kind": "number|name|place|date|equipment|unit|event", "value": "<exact figure if a number, else the key detail>"}
]

STORY:
{story}
"""

# ---------------------------------------------------------------- 2. story

ANGLE_PROMPT = """Read the facts below and find the story's ENGINE before writing anything. You are planning ONLY - no prose, no sentences for the video yet.

Answer exactly:
- "reversal": the surprising gap at the heart of the story, in one short clause with no numbers ("a tiny crew made a superpower see an army that was not there").
- "fooled": who believed what - one short clause ("the Germans believed they faced an army of 30,000").
- "hook_ticket": the single most shocking thing a first sentence could state that is 100% true per the facts (one short clause).
- "contrast_pairs": exactly 3 number/twist contrasts ordered so the third is the most damning, e.g. ["1,100 men vs a vision of 30,000", "...", "..."].
- "beat_order": 4 to 7 short rising-tension labels for the scenes, the last one the reveal, e.g. ["open on the leap of faith", "show the tiny crew", "climax on the believed army", "reveal smoke and cloth"].

Return ONLY JSON:
{{
  "reversal": "...",
  "fooled": "...",
  "hook_ticket": "...",
  "contrast_pairs": ["...", "...", "..."],
  "beat_order": ["...", "...", "..."]
}}
No markdown, no prose.

FACTS:
{facts}"""

GOLD_PATTERNS = """GOLD PATTERNS TO IMITATE (imitate the FORM, never reuse the topic or any example's wording)
- Hook: "Imagine an army that didn't exist. It fooled an entire nation." / "The safest ship ever built sank on its maiden voyage."
- Number-kill beat: "1,100 men. Germany saw 30,000." - two numbers, no filler, the gap IS the sentence.
- Escalation beat: a short image first, then the scale that makes it terrifying: "Inflatable tanks in the dark. From 500 yards, they were real."
- Closer: a question that makes someone type in the comments: "What else is out there we are not being told?" - exactly ONE question line, never a stack."""

STORY_PROMPT = """Write a short spoken narration for a faceless war-story video. A viewer is mid-scroll with their thumb 3 cm from the next video - the first sentence decides whether they stop. Your job is to make them stop and keep them watching.

STRUCTURE LOCK (ground truth - do not deviate)
The reversal and the "fooled" relationship below are the engine of the story. Every sentence must serve them. Do not invent a competing frame (no "they lost", no "they hid tanks", no "they infiltrated").
- REVERSAL: {reversal_hint}
- FOOLED: {fooled_hint}
- HOOK TICKET: the opening line must grow from this fact - {hook_ticket_hint}
- CONTRAST PAIRS in order, save the reveal for near the end: {pairs_hint}. You turn each pair into a NARRATED CONTRAST in your own words (two numbers or two images side by side). Never write the hint tokens themselves - no "vs", no "X instead of Y" verbatim, no "instead of reality" anywhere.
- BEAT ORDER: {beats_hint}

RULES
- One sentence per line. 8 to 12 lines total - never fewer than 8, never more than 12. Every sentence is one short, spoken thought (3-7 seconds when read aloud).
- The NARRATOR VOICE is a confident, slightly unhinged war-storyteller who talks like a real person, never like Wikipedia.
- LINE 1 is the hook: one specific, surprising image or number built from the hook ticket. Never a greeting, never "today we will", never "did you know", never a calm fact. Create an open loop immediately.
- Use the EXACT numbers from the facts. The number is the story - never "lots of them", "countless", "half the army", "thousands" when the real figure is known. Feature TWO big numbers close together at least once (the contrast beats are the scene).
- Short, staccato beats for quiet moments. One line can carry the biggest moment with caps or "!!" (AT MOST one line).
- FINAL LINE: the closer. Use the contrast-pair reveal or one earned question that points back at the hook. AT MOST ONE question line in the whole script - a stack of questions is a fail. Do not invent a threat the facts never established.
- ABSOLUTE RULE: do not invent ANY fact, number, place, event, or twist that is not in the FACTS list - including inside the hook and the closer. Every sentence must trace back to the facts.

{GOLD_PATTERNS_BLOCK}

FACTS:
{facts}

Output ONLY the narration: one sentence per line, no numbering, no headings, no JSON, no explanations."""

# ---------------------------------------------------------------- 3. theme
#
# Theme is applied LINE BY LINE: one bounded sentence per call, re-voiced in
# a persona. Unlike the story stage (where example sentences get parroted into
# the facts), a theme is a FIXED persona across every story - so a catchphrase
# lexicon is a feature, not a leak. Exemplars below use alien topics so the
# model learns the SENTENCE SHAPE of the voice, not words it could paste in.

_ARNOLD_BLOCK = """Re-voice ONE narration sentence so it sounds like ARNOLD SCHWARZENEGGER: a battle-hardened commander reviewing the operation.

HOW HE TALKS: plain everyday words, short commanding clauses, one clear thought per line. He states hard judgments with total calm. He never rambles and never goes soft.

CATCHPHRASE BANK - use AT MOST one per line, only where the sentence's meaning genuinely fits it:
- "Listen close:" to front a number or a lesson
- "Let me tell you something." to open mid-story emphasis
- "That is how it is done" / "That is how you do it" after a show of skill
- "It was over before it started." / "A lost battle from day one." ONLY for a line that actually shows a failure
- "Get to the point." only as a self-interruption
- "Some things never leave the battlefield." for the closer
Otherwise write plain commander speech - do NOT force a catchphrase into every line.

EXEMPLARS - imitate the SHAPE (open, pace, word choices); the topics are foreign and must never appear in your output; each shows an EXACTLY-ONE-SENTENCE re-voice:
- Opening hook: "The safest ship ever built sank on its maiden voyage." -> "Hey, come in tight - the safest ship ever built went down on the first crossing."
- Number beat: "The dam held back 40,000 tons of water." -> "Listen close: 40,000 tons of water, and that wall held - that is what training does."
- Failure judgment: "The unit was caught in the open and wiped out." -> "Caught out in the open and wiped out - a good plan, badly timed, and that is how you lose battles."
- Closer: "Historians still argue about who gave the order." -> "Some things never leave the battlefield - and who gave that order is one of them."

STRICT
- Keep THIS sentence's facts and numbers EXACTLY. Never add an outcome (won, lost, infiltrated, failed) the sentence does not show.
- Output EXACTLY ONE sentence - one period, question mark, or exclamation at the end. Never two sentences.
- Never bend a catchphrase into a lie about the facts.
- Never say "okay" or "ok". Never gym or bodybuilding language."""

_TRUMP_BLOCK = """Re-voice ONE narration sentence so it sounds like DONALD TRUMP: a bragging storyteller ranking a huge failure.

HOW HE TALKS: short punchy sentences, very simple words, every claim stated with total confidence as a fact. He ranks things - each new point is bigger, worse, or more incredible than the last.

CATCHPHRASE BANK - use AT MOST one per line, only where the sentence's meaning genuinely fits it:
- "Believe me" / "It's true, believe me" after a big claim
- "Huge." / "Tremendous." / "Incredible." as a one-word verdict (max twice per whole narration)
- "They said it couldn't be done." for a line showing doubt being beaten
- "Big, big mistake." for a clear blunder line
- "And it wasn't even close" to slam an outcome
- "Think about that one." for the closer
Otherwise write plain punchy claims - do NOT force a catchphrase into every line.

EXEMPLARS - imitate the SHAPE; the topics are foreign and must never appear in your output; each shows an EXACTLY-ONE-SENTENCE re-voice:
- Opening hook: "The safest ship ever built sank on its maiden voyage." -> "This was going to be the greatest ship of all time - and it went down on the first voyage."
- Scale claim: "The dam held back 40,000 tons of water." -> "Forty thousand tons of water - a massive, massive number, and the dam held it."
- Failure ranking: "The unit was caught in the open and wiped out." -> "Caught in the open and wiped out - honestly they never had a chance, a big mistake from the top."
- Closer: "Historians still argue about who gave the order." -> "And the greatest part? They still cannot figure out who gave that order."

STRICT
- Keep THIS sentence's facts and numbers EXACTLY. Never add an outcome (won, lost, infiltrated, failed) the sentence does not show.
- Output EXACTLY ONE sentence - one period, question mark, or exclamation at the end. Never two sentences.
- Never bend a catchphrase into a lie about the facts.
- Never say "okay" or "ok". Catchphrases at most once per line."""

THEME_LINE_PROMPT = """Re-voice the narration sentence below in the given voice. Output ONLY the re-voiced sentence(s).

VOICE:
{moves}

START OF SCRIPT (line 1, hook): {first_flag}

STORY CONTEXT (for continuity only - re-voice ONLY the SENTENCE below, not the context):
{prose}

SENTENCE TO RE-VOICE:
{sentence}

Output ONLY the re-voiced sentence(s). No numbering, no headings, no JSON."""

THEMES = {
    "arnold": _ARNOLD_BLOCK,
    "trump": _TRUMP_BLOCK,
}

# ---------------------------------------------------------------- 4. json

SCRIPT_JSON_PROMPT = """Convert the narration into structured JSON lines for a video editor.

Rules
- Split exactly on sentence boundaries: each line is ONE spoken sentence, in the original order. Keep the words verbatim when possible.
- NEVER split a single sentence into two lines on a comma, dash, or colon. "Hey, listen up. We had..." is ONE line, not two. A fragment like "Involving..." that is not a full sentence must be joined into its sentence.
- 8 to 12 lines.
- Each line object:
[
  {"text": "<sentence>", "beat": "hook|setup|escalate|payoff|closer", "tone": "normal|drop|shout"}}
]
- beat labels: line 1 = hook; later escalation = escalate; the single highest-moment line = payoff (put it near the end); final line = closer; the rest = setup/escalate as appropriate.
- tone: normal for most; drop for any lingering/quiet line (ellipsis or short staccato beats); shout for the ONE line written in caps or ending in "!!".

NARRATION:
{prose}

Return ONLY the JSON array. No markdown, no prose."""

# ---------------------------------------------------------------- 5. queries

QUERY_PROMPT = """You craft image-search queries for a faceless war-story short. Each narration line currently included with the story context.

For EVERY line provide TWO or more DISTINCT image queries that a stock-video site would return good footage for.

Rules
- 2 to 5 words per query. Visual and concrete, not abstract: a search like "ww2 inflatable tank field" is good; "deception in war" is useless.
- Pull from the story's facts for specificity: the era, the place, the equipment, the unit, the terrain.
- Vary the queries so each one targets a different shot (wide scene / close detail / objects / crowd / sky).
- No names or exact numbers inside queries. No modern objects if the story is historical.
- Topics to fall back on if a line has no visual anchor: need a list of 6 generic war-footage queries usable anywhere.

Return ONLY a JSON array, one object per line:
[
  {"line": 1, "queries": ["ww2 inflatable tank field", "soldiers night march", "..."]}
]

TOPIC: {topic}
STORY: {story}
LINES (JSON):
{lines_json}

Return ONLY the JSON array. No markdown, no prose."""