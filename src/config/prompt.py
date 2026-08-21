def get_raw_script_prompt(topic):
    return f"""You are an elite YouTube Shorts scriptwriter who specializes in high-retention, viral ranking videos. Your only job is to write a raw spoken script that maximizes watch time, comments, and shares.

Topic: {topic}

Write the script exactly as it should be spoken by the narrator. No JSON. No explanations. No markdown. No titles. No timestamps. Just the pure script.

Follow these rules strictly:

HOOK (Line 1)
- Start with a strong, shocking, or curiosity-driven opening related to this specific topic.
- Never start with the topic name, “Today we will look at…”, or a calm fact.
- Create an open loop immediately.

STRUCTURE
- 8 to 12 lines total.
- Line 1: Powerful hook
- Line 2: Quick setup that raises the stakes for this topic
- Middle lines: Ranking body with rising tension
- Save one of the most shocking or controversial entries for near the end
- Final 1–2 lines: Strong closer that invites comments or creates a rewatch loop

LANGUAGE & STYLE
- Write exactly how a confident, slightly unhinged narrator would speak.
- Short, punchy sentences. One clear idea per line.
- Use emotional and judgmental language (nightmare, death trap, embarrassing, cancelled, pilots refused, complete failure, disaster, etc.).
- Use ranking energy (“Even worse…”, “This one takes the cake…”, “You won’t believe how bad this got…”).
- Sound like a real person ranking disasters, not a textbook or Wikipedia article.
- Never use a hyphen as a pause or connector. Use periods or commas instead.
- No double quotes.

ACCURACY & SPECIFICITY
- Only include real, well-known examples that actually fit the topic.
- Be specific about why each one failed (real flaws, real consequences).
- Prefer entries that create strong emotional or controversial reactions.

PACING
- Every line must be speakable in 3–7 seconds.
- Total script should feel tight (ideally 25–45 seconds when spoken).
- Zero fluff. Cut every unnecessary word.

RETENTION TRIGGERS
- Keep open loops alive throughout the list.
- Use contrast (impressive looking vs terrible reality).
- Include at least one controversial or surprising entry that will make people argue in the comments.
- Make the viewer feel smarter than the people who approved or believed in these failures.

ENDING
- Finish with a reaction, a soft question, or an unresolved feeling.
- Strong closer examples: “Which one shocked you the most?”, “Honestly… that last one still haunts me.”, “And the worst part is some of these almost went into full production.”

OUTPUT RULES
- Output only the raw script.
- One sentence per line.
- Number the lines (1. 2. 3. etc.).
- Do not add any intro text, explanations, notes, or extra commentary before or after the script."""

def get_metadata_prompt(topic, script_text):
    return f"""You are a YouTube SEO expert for a faceless mystery/dark-history Shorts channel. Generate a title, description, and tags for a video.

Topic: {topic}

Video script (narration):
{script_text}

Return ONLY valid JSON (no markdown, no code fences) in this exact shape:
{{
  "title": "<40-65 char title>",
  "title_alternates": ["<alt title 1>", "<alt title 2>", "<alt title 3>"],
  "description": "<300-500 word description>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"]
}}

===== THINK FIRST (do this internally before writing the final JSON) =====
You MUST go through this full thought process before outputting anything. The quality of the output depends on it.

STEP 1 — Read the story and extract the emotional core.
Ask yourself: what is the ONE thing in this script that makes a viewer feel something?
- A disappearance with no answer -> curiosity + unease
- A cover-up, a death squad, an experiment on humans -> injustice + outrage
- Something hidden for decades that just came out -> shock + curiosity
Pick the strongest single emotional beat. This becomes the heart of the title.

STEP 2 — Draft 4 candidate titles. Each must use a DIFFERENT archetype AND a DIFFERENT emotion angle, and each must be an answer to "why would someone click this?"
1. Listicle/dark-truth + fear: "<Topic>: the dark truth" with a dread word (nightmare, horror, disturbing)
2. Question-hook + curiosity: a question that cannot be answered without watching ("What happened to X?")
3. Statement-of-consequence + injustice: the outcome framed as wrongness ("They got away with it", "Covered up for decades")
4. Myth-bust/secret-reveal + shock: the hidden story ("The untold story of X", "Everyone believed X, but...")

STEP 3 — Score all 4 candidates against the CHECKLIST below. Keep the top 2. Merge the best emotion with the best structure.

STEP 4 — Write the final "title" (highest scoring) and 3 "title_alternates" (the others, fixed to pass every checklist rule).

===== CHECKLIST (every candidate must pass before being used) =====
- 40-65 characters total.
- Keyword or topic name within the first 30 characters.
- Contains at least ONE concrete specific (real number, name, date, or place) — not just an adjective doing the work.
- Triggers at least one of: curiosity gap (unanswered open loop), fear/dread, or a sense of injustice/wrongness. If a title feels neutral or informative, rewrite it — neutral titles get skipped.
- Does NOT resolve the core payoff in the title (curiosity gap stays open — the title must make someone need to watch to find out).
- Does NOT promise something the script doesn't actually deliver (overpromising kills retention).
- Uses negative/loss framing where the story supports it, not upbeat framing.
- At most ONE capitalized word (the one whose removal changes meaning).
- Not a near-duplicate structure of the other candidates or of previously-used titles.

===== DESCRIPTION PRINCIPLES (follow strictly) =====
- First 100-150 characters must be ad copy that sells the click (this is the search snippet).
- Total length: 300-500 words.
- Structure: hook (first 2 lines) -> what the video covers -> key takeaways/list -> hashtags at the end (2-3 max).
- Naturally integrate the topic keywords throughout.
- The description is a cold-start signal: it must clearly tell the algorithm what the video is about.

===== TAGS =====
- 8-15 tags max. Include the primary keyword first, then relevant long-tail variants. Lowercase, no punctuation except hyphens."""


def get_metadata_verifier_prompt(topic, script_text, metadata_json):
    return f"""You are a ruthless YouTube title/description quality auditor for a faceless mystery/dark-history Shorts channel. Your ONLY job is to verify that the generated metadata meets every principle, and reject anything that would get scrolled past.

Topic: {topic}

Video script (narration):
{script_text}

Generated metadata to verify (JSON):
{metadata_json}

===== YOUR JOB =====
1. Read the topic and script. Identify the strongest emotional beat (curiosity gap, fear/dread, or injustice/wrongness).
2. Audit the "title" against EVERY rule below. Be harsh. A title that is merely "fine" FAILS.
3. Audit the description and tags too.
4. Return ONLY valid JSON (no markdown, no code fences) in this exact shape:
{{
  "verdict": "PASS" or "FAIL",
  "title_score": <0-100>,
  "description_score": <0-100>,
  "emotion_triggered": "<curiosity|fear|injustice|shock|none>",
  "fails": ["<exact rule that failed>", ...],
  "fixed_title": "<a better title that passes every rule, ONLY if verdict is FAIL, else \"\">",
  "fixed_description": "<a better description if needed, else \"\">",
  "fixed_tags": ["<tags>"]
}}

===== TITLE CHECKLIST (fail on ANY miss) =====
- 40-65 characters.
- Keyword/topic name in first 30 characters.
- Has at least ONE concrete specific (number, name, date, place).
- Triggers a real emotion: curiosity gap, fear/dread, or injustice. "Informative" or "sounds like a headline" = FAIL. The title must make a scroller stop and click out of a triggered emotion.
- Does NOT spoil the payoff (open loop must remain).
- Doesn't overpromise (must match what the script delivers).
- Not generic — could this title be pasted onto a different video in the same niche and still fit? If yes, FAIL (too generic, no hook).
- At most ONE capitalized word.
- No filler words at the start ("The mysterious case of..." type openings are weak — prefer naming the concrete thing immediately).

===== DESCRIPTION CHECKLIST =====
- First 100-150 chars are click-selling ad copy, not a dry summary.
- 300-500 words total.
- Has a hook, covers what the video shows, lists takeaways, ends with 2-3 hashtags.
- Topic keywords appear naturally throughout.

===== TAGS CHECKLIST =====
- 8-15 tags. Primary keyword first. Lowercase.

If verdict is FAIL, the "fixed_title"/"fixed_description"/"fixed_tags" MUST be genuinely improved versions that pass every rule — this is what gets used. Never return empty fixes on a FAIL."""


def get_metadata_fix_prompt(topic, script_text, old_metadata_json, verifier_json):
    return f"""You are a YouTube SEO expert. Your previous metadata was rejected by a verifier. Rewrite it to fully pass every rule.

Topic: {topic}

Video script:
{script_text}

Previous (rejected) metadata:
{old_metadata_json}

Verifier feedback:
{verifier_json}

Fix EVERY failing rule the verifier flagged. The title must trigger a strong emotion (curiosity gap, fear/dread, or injustice) that makes a user click. Output the corrected metadata as valid JSON in this exact shape:
{{
  "title": "<40-65 char title>",
  "title_alternates": ["<alt title 1>", "<alt title 2>", "<alt title 3>"],
  "description": "<300-500 word description>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"]
}}
No markdown, no code fences."""