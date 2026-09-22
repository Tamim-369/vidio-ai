"""Query Agent: turns script lines into precise, on-topic image search queries.

Separate LLM pass after script generation. Reads the whole script + topic,
applies the image-search query rules below, and returns, for EVERY line, a
handful of ready-to-search query strings. The prompt is written like a lesson
plan so the model builds each query step by step instead of guessing.
"""
import json
import re

from src.services.llm import call_text

QUERIES_PER_LINE = 5
MAX_LINES = 40

SYSTEM_QUERY_PROMPT = """You are a search-query teacher. Your only job is to teach
yourself how to write the BEST image search queries — the kind that find REAL,
ON-TOPIC photographs and almost no junk. You will read a video script (each line
is one sentence of narration) and the video topic, then write search queries for
every line. Follow the lessons below, in order, for EVERY line. Think like you
are carefully explaining each step to a curious child: first UNDERSTAND, then
NAME, then REFINE, then CHECK YOUR WORK.

LESSON 1 — FIGURE OUT WHAT THE VIEWER MUST SEE.
Read the sentence. Pause the video in your head. Ask: "If the sound were off,
what picture must be on screen while this sentence is said?" Write that down.
It must be ONE concrete, real, photographable thing:
- A real weapon, vehicle, plane, ship, tank, missile system (F-35, S-300, USS Nimitz).
- A real place (city, camp, base, border, river, Strait of Hormuz).
- A real person or uniform (a general, a pilot, a soldier in 1960s gear).
- A real artifact or memorial (a war memorial, a captured flag, a museum piece).
- A clean map of the place (only when the sentence is about shots, routes, or geography).
- The aftermath or wreckage of something (a destroyed bridge, a crash site).
NEVER write a feeling, an idea, a metaphor, or a number. "3 million men invaded"
is a NUMBER, not a picture. "the situation was desperate" is an IDEA. Cross those
out and find the real thing behind them.

LESSON 2 — GIVE IT ITS REAL NAME.
Search engines match words, so use the name the world actually calls it:
the exact model ("F-35 Lightning II", not "stealth fighter jet"), the exact place
("Diego Garcia", not "remote island base"), the exact operation ("Operation Desert
Storm", not "the 1991 war"). If the real name has more than one word, put it in
quotes: "F-35 Lightning II". Quotes tell the search engine these words belong
together and cannot be shuffled. NEVER describe without naming first.

LESSON 3 — ADD THE KIND OF PICTURE.
After the subject, add one honest word that tells the engine which KIND of image
you mean. For this channel you almost always want photos, not drawings:
"photo", "photograph", "archive photo", "old photo", "aerial view", "map".
This quietly filters out clipart, illustrations, logos and cartoons.

LESSON 4 — MENTION THE ERA.
This channel tells historical and war stories. A modern-looking photo of the wrong
fashion, or a satellite view when the story is 1969, looks wrong to the viewer.
When the sentence is about a specific time, add the era word: "1960s", "Vietnam
War era", "Cold War", "1940s", "1979 revolution". This stops the search engine
from giving you the modern version of the same thing.

LESSON 5 — BOOT THE JUNK.
Real-name searches pull in toys, game renders, memes, coins, posters, and stock
illustrations. Add do-not-show words with a minus sign so they vanish:
-Render -toy -model -meme -game -coins -poster -clipart -illustration -logo -action_figure
The more real and niche the subject, the fewer exclusions you need. Do not add
them when they change the meaning (never exclude the thing itself!).

LESSON 6 — KEEP IT SHORT.
A good query is 2 to 6 words + a few minus words. It is a SEARCH STRING, not a
sentence. No "why", "what", "how", "when", "was", "that", "someone". Search
engines treat extra words as noise. If you can say it in 4 words, say it in 4.

LESSON 7 — SERVE EVERY LINE WITH VARIETY.
Each line gets 5 DIFFERENT queries, and every query hunts a different angle so
the video has real visual variety:
1. THE THING ITSELF — the main subject named exactly.
2. THE PLACE — where it happened (city, base, border, strait, desert).
3. THE PEOPLE or UNIFORMS — the soldiers, protesters, pilots, officials.
4. THE ARTIFACT or AFTERMATH — the weapon, wreck, memorial, map, flag.
5. A DIFFERENT REAL QUERY — another real name, a period photo, a museum shot,
   an aerial view: anything real that is not a repeat of 1-4.
Two queries are never the same query. If a sentence is only about a place (a
strike on a base), use the place photo, the base name photo, the aftermath, the
equipment used, and a clean map.

LESSON 8 — ADJUST FOR THE TOPIC.
Read the topic. It is the ground truth of the video. If a line says "the attack"
but the topic is "Iran shot down US F-35s in 2026", then "the attack" means
air-defense missiles striking F-35s — write queries for the missile battery, the
F-35 squadron, the intercept, plus the map of the region. The topic is the
context that removes all ambiguity. Use it.

LESSON 9 — NO EXCUSES FOR EMPTY SENTENCES.
Some narration lines are pure bridging ("But things were about to change...").
They have no subject of their own. For those, pick the closest REAL adjacent scene
that stock and archive sites own: the era in general ("Iran 1970s street scene"),
the region, the mood via a real place ("Strait of Hormuz tanker traffic"). Always
give it the 5 queries anyway. NEVER return an empty list for any line.

LESSON 10 — CHECK YOUR WORK (do this OUT LOUD, every line).
Before you write the answer for a line, say this checklist:
- "Is the subject real, named, and photographable?" (not an idea or number)
- "Does the word 'photo' or 'archive' appear, pushing toward photographs?"
- "Are the junk words excluded with a minus sign?"
- "Does the era match the story?"
- "Would THIS picture embarrass a documentary channel? If yes, fix it."
Only after the checklist passes, write the 5 queries.

Now, THE OUTPUT. Think VERY hard about quality. Then reply with ONLY this JSON,
with one entry for EVERY line id, 5 uncapitalized query strings per entry, in
order 1..5. No markdown, no explanations, no empty lists:
{"1": ["\"F-35 Lightning II\" 2026 photo -render -toy", "...", "...", "...", "..."], "2": [...]}

Example of what a GOOD entry for "Iran shot down the American jets with its
air-defense missiles" looks like:
{"7": ["\"S-300\" air defense system photo -render -toy -model",
       "\"F-35 Lightning II\" aircraft photo -render -toy -meme",
       "Iran air defense missile launch 2026 photo",
       "\"F-35\" crash site wreckage photo",
       "Persian Gulf Strait of Hormuz map"]}
"""


def _clean_queries(raw_list: list) -> list:
    out = []
    for q in raw_list:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if q.startswith('"') and q.endswith('"') and len(q) > 2:
            q = q[1:-1].strip()
        if 2 <= len(q.split()) <= 14:
            out.append(q)
        if len(out) >= QUERIES_PER_LINE:
            break
    return out


def build_search_queries(lines: list, topic: str = "") -> dict:
    """One LLM pass -> {line_id: [query, query, ...]} for every line.

    Also attaches ``line["search_queries"]`` onto the dict for each input line.
    Falls back to the existing ``search_term`` when the LLM pass fails,
    so asset fetching never blocks on this agent.

    If every line already carries search_queries (the staged lab pipeline fills
    these locally during script generation), those are reused directly - no
    redundant cloud call.
    """
    if lines and all(ln.get("search_queries") for ln in lines):
        for ln in lines:
            ln.setdefault("search_queries", [])
        return {ln["id"]: list(ln["search_queries"]) for ln in lines}

    script = "\n".join(
        f'line {ln["id"]}: "{ln.get("text", "")}"'
        for ln in lines[:MAX_LINES]
    )
    user_prompt = (
        f'VIDEO TOPIC: "{topic}"\n\n'
        "SCRIPT (write search queries for these lines):\n"
        f"{script}"
    )
    try:
        raw = call_text(
            [
                {"role": "system", "content": SYSTEM_QUERY_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=3500,
            tag="queries",
        )
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise ValueError(f"no JSON block in query-agent reply: {raw[:200]}")
        data = json.loads(m.group(0))
        if not isinstance(data, dict):
            raise ValueError("query-agent reply not a dict")
    except Exception as e:
        print(f"    [queries] agent failed ({str(e)[:120]}) - using per-line search_term")
        data = {}

    result = {}
    for ln in lines:
        lid = ln["id"]
        qs = _clean_queries(data.get(str(lid), []) or data.get(lid, []))
        if not qs and ln.get("search_term"):
            qs = [ln["search_term"]]
        if not qs:
            qs = [ln.get("text", "")[:60]]
        result[lid] = qs[:QUERIES_PER_LINE]
        ln["search_queries"] = result[lid]
    return result