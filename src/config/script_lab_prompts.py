"""Prompt set for the staged script-production pipeline (script_lab).

Stage order:
  1. STORY  - turn the raw source into a short, detailed, easy-to-follow story,
              written directly in the narrator's speaking style in ONE pass
  2. AGENT  - turn the story into structured JSON script lines + image queries

The interestingness of the script comes from the STORY-layer instructions and
the character speaking styles (my design), NOT from trusting a big model. The
theme re-voice stage is gone: the speaking style is injected into this prompt,
so one narrow well-bounded pass produces the voiced story.
"""

# ---------------------------------------------------------------- 1. story

STORY_PROMPT = """Read the material below and write a short, detailed story out of it. The story must be easy to understand for a normal adult - not childish, not a dry technical report.

{style}

WHAT THE STORY IS FOR
- A viewer with no background in this topic should be able to follow exactly what happened, to whom, where, and what it meant.
- This is a STORY, not a summary or a data dump. It needs a clear arc: who, what happened, how it unfolded, and what it led to.

TELL IT WITH FEELING - YOU ARE NOT TRANSCRIBING
- You are NOT relaying the material like a news bulletin, and you are NOT tidying up some existing text line-by-line. YOU just found out this story and it shook you: you are telling it as the person it shocked, amazed, horrified, or moved.
- Let that emotion bleed into the telling - react, judge, gasp, rank, linger on the detail that hit you hardest. The names, numbers, dates, and places stay exactly as they are; but the voice saying them is a human being hit by them, not a textbook.
- Keep the order of events clear: what happened first, then next, then how it ended.

MAKE IT STOP-SCROLL (THIS IS THE MOST IMPORTANT RULE)
- THE FIRST LINE MUST BE A HOOK that literally interrupts the viewer's thumb: address them OUT LOUD and stop them, or make a scroll-stopping promise that makes them NEED to know. Each style has its own opener flavor (see above) - use THAT, freshly worded, never the example words verbatim.
- It fits in 1-2 short lines, then the very next beat drops the single strangest, most specific fact from the material.
- ONE READY-TO-SAY SENTENCE PER LINE. Every line is ONE COMPLETE SENTENCE that can stand alone: it has its own subject and verb and ends with a full stop or "!". This text is read aloud as-is, so each line must sound like a finished spoken sentence.
- TWO IDEAS = TWO SENTENCES. Whenever you find yourself about to glue a second idea onto a sentence - whether with a dash, a semicolon, or a comma inside one line - STOP. You are not done; you are about to write two thoughts in one line. Write TWO separate complete sentences instead, each on its own line. A dash inside a sentence is ALWAYS a mistake: it means "two sentences here".
- The HOOK is its own short complete sentence on the first line. The very next fact is its own complete sentence on the second line. Never join them.
- A line must never begin again where the previous line could already have ended. If the last word could be a full stop, make it one.
- FORBIDDEN first-line formulas that instantly lose the viewer: starting with a year or date ("In 1933..."), starting with the topic name, "This is a story about...", "The history of...", a dry summary, or naming the country/regime before anything happens in it. An interrupt or a scroll-stopping promise is none of these.
- The first 1-2 lines must raise a question the viewer cannot answer without watching: "how could they possibly do that?", "did they survive?" Do NOT answer it immediately - leave it dangling and resolve it by the end.
- Put a STAKES line in the first two sentences: what was on the line (lives, a secret, a nation, survival). The viewer must feel something could go wrong.
- Every line should pull toward the next. If a sentence can be cut without losing tension, cut it or fold it into another.
- Show, don't tell: pick the concrete, picture-able detail (a number, an object, a place, a sound) over an adjective.
- Build tension to a peak: the last big reveal or the darkest moment lands near the end, after which ONE short closing line can hit.

HOW TO HANDLE TECHNICAL DETAIL
- Include technical elements ONLY where the story needs them to make sense - enough to identify what happened and keep the context clear.
- Cut technical detail that does not help explain the story. If removing it would not confuse anyone, leave it out.
- Never change or drop names, numbers, dates, or places that the story depends on - keep them exact.

HOW TO WRITE
- Short but detailed: every sentence should carry real information about what happened.
- NEVER copy a sentence, clause, or phrase from the material. Say every fact in your own words.
- Every sentence must say something NEW. If a point is already made, do not repeat it. No echoing or rewording the same fact twice.

LENGTH
- 8 to 12 sentences. One sentence per line.

MATERIAL:
{story}

Output ONLY the story: one sentence per line, no headings, no JSON, no explanations."""

# ---------------------------------------------------------------- 2. agent

SCRIPT_AGENT_PROMPT = """You are the script editor for a faceless short video. Convert the narration below into the final structured script: one JSON object per spoken line, in the original order.

Rules
- Split exactly on sentence boundaries: each line is ONE spoken sentence, in the original order. Keep the words verbatim when possible.
- NEVER split a single sentence into two lines on a comma, dash, or colon. "Hey, listen up. We had..." is ONE line, not two. A fragment like "Involving..." that is not a full sentence must be joined into its sentence.
- If the narration has glued two ideas into one sentence ("...won the war - it used fake tanks, not guns"), REWRITE that line into TWO separate complete sentences so each idea is understood alone. Change a dash or semicolon between two ideas into a full stop.
- 8 to 12 lines.
- Each line object:
[
  {"text": "<sentence>", "beat": "hook|setup|escalate|payoff|closer", "tone": "normal|drop|shout"}
]
- beat labels: line 1 = hook; later escalation = escalate; the single highest-moment line = payoff (put it near the end); final line = closer; the rest = setup/escalate as appropriate.
- tone: NOT every line is "normal" - vary it like a real video. normal for most; drop for any lingering/quiet line and for the line right before the payoff or closer; shout for the ONE loudest line (the payoff or the highest-moment line). The hook should be punchy, not flat.

NARRATION:
{prose}

Return ONLY the JSON array. No markdown, no prose."""

# ---------------------------------------------------------------- 3. queries

QUERY_BUILDER_PROMPT = """You are the image-search engineer for a documentary short-video channel. You get the VIDEO TOPIC and one line per narration sentence. Your job: for EVERY line write image-search queries that find REAL, ON-TOPIC photographs and footage - and reject anything that does not belong to this topic.

WORK FROM THE TOPIC, NOT THE SENTENCE WORD-FOR-WORD.
- The TOPIC is the ground truth of the whole video. Every query must be about the TOPIC.
- A narration line is the VOICE, not a camera direction. It can use slang, nicknames, vague words, or metaphors ("Hitler's buzzsaw", "the attack", "that commander"). You must figure out what REAL, NAMED, PHOTOGRAPHABLE thing that line means IN THIS TOPIC, then search for that thing - never for the sentence's literal words.
- EXAMPLE. TOPIC: "German soldier best weapon". A line says "Hitler's buzzsaw was an amazing gun." Do NOT search "buzzsaw" or "amazing gun". The real subject this topic points to is the MG42 machine gun (nicknamed Hitler's Buzzsaw). Search "MG42 machine gun".
- EXAMPLE. TOPIC: "F-35 shootdown 2026". A line says "their missiles found the jets." Do NOT search "their missiles found the jets". Search the real thing: the air-defense missile system and the F-35.
- RESOLVE FIRST: for each line, quietly decide the REAL SUBJECT to film for THIS TOPIC before writing any query. If the speaker used a nickname, a pronoun, a metaphor, or a vague phrase ("they", "it", "that day"), translate it into the concrete named subject from the topic.
- TOPIC LOCK: if a line has no real on-topic subject (a pure bridging sentence like "But things were about to change"), do NOT search the sentence text. Search the closest REAL adjacent scene the topic owns - the era, the place, the people, the equipment, the artifact. NEVER return a query that has nothing to do with the topic.

WRITE QUERIES THE WAY SEARCH ENGINES MATCH.
- NAME the real thing exactly: exact model/name/place/operation ("MG42", "USS Nimitz", "Operation Desert Storm"). Multi-word real names go in quotes so the engine keeps them together: "MG42 machine gun".
- A number, idea, feeling, or metaphor is NOT a picture ("3 million men", "the situation was desperate"). Find the real object behind it.
- Add ONE kind-of-picture word: photo, photograph, archive photo, old photo, aerial view, map, close-up. This filters out illustrations and cartoons.
- Add the ERA when the story is historical ("1940s", "WWII", "Cold War era") so you get the period, not the modern version.
- EXCLUDE JUNK with minus signs: -render -toy -model -meme -game -poster -clipart -illustration -logo -action_figure -news -press -watermark. Add more per subject if needed; never exclude the real subject itself.
- SEARCH STRING, not a sentence: no filler words like "why/what/how/was/the/that/a". Use exactly the words that pinpoint the real subject - and never make up words, never guess a name, never invent facts. Use only words the TOPIC and the line actually support. There is no word limit: use as many words as the real subject needs.
- COPYRIGHT-SAFE, NOT NEWS: this channel can only use free/archival imagery. NEVER write a query that pulls a modern news-wire or press photo (Reuters, AP, AFP, Getty - those are copyrighted and unlicensed). For historical topics prefer words that retrieve ARCHIVAL or PUBLIC-DOMAIN material: "archive photo", "period photograph", "museum archive", "public domain photo", plus the ERA word ("1940s", "WWII"). Do not write "news", "today", "latest", "press conference", "handout", or any reporting word.

EVERY LINE GETS 3 DIFFERENT QUERIES - each a DIFFERENT angle for visual variety:
1. THE THING ITSELF - the main named subject.
2. THE PLACE OR THE PEOPLE - where it happened (city, base, the soldiers, the uniforms).
3. A DIFFERENT REAL ANGLE - the artifact aftermath, an aerial view, a museum piece - not a repeat of 1-2.
Two queries are never identical. For pure bridging lines, cover the closest real topic scene: the era in general, the region, the equipment.

FIRST LINE IS THE THUMB-STOPPER: its queries must produce the strongest, most striking, most on-topic images of the whole video - the frame that makes a viewer stop scrolling.

TOPIC CHECK (say it for every query): "Is this about the TOPIC? Is the real, named, photographable subject there? Would this picture embarrass a documentary channel? If yes, fix it." Any query that is NOT about the topic must be replaced with one that IS about the topic. Never leave a line empty.

VIDEO TOPIC:
{topic}

NARRATION LINES (one per line, number them exactly like this):
{script}

Reply with ONLY this JSON - one key per line number, 3 query strings per entry, no markdown, no explanations:
{"1": ["query 1", "query 2", "query 3"], "2": [...]}"""