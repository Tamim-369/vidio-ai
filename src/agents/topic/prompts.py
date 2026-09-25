"""All LLM prompts for the topic agent.

Two brainstorm paths plus the source-discovery prompt live here so the model
calls in ``gemini.py`` / ``miner.py`` stay pure I/O. The source-inventory
prompt (TOPIC_SOURCE_PROMPT) is the manual research aid that produced
``sources.py`` — kept here for reference.
"""

TOPIC_SOURCE_PROMPT = """You are a researcher finding the BEST sources for viral YouTube Shorts topics.

NICHE:
  Faceless shorts about the dark edges of the world wars and the legends they left behind:
  1. Crazy human and animal experiments run by the great powers in WW1/WW2
     (Unit 731, Nazi medical experiments, chemical/radiation tests on prisoners
     and soldiers, secret labs using living people as specimens).
  2. Scary stories and legends rooted in real history: curses, hauntings,
     urban legends, cover-ups, bizarre-ish true events the archives barely talk about.
  3. Epic true WW1/WW2 stories: last stands, impossible rescues, outnumbered
     units, the human moments inside the biggest wars ever fought.

GOAL: Name the TOP sources that most reliably produce VIRAL-READY stories in this niche.

For each source, real-world sources ONLY — things that actually exist and can
be fetched programmatically. Prefer raw feeds over walls of text:
  - RSS/Atom feeds of history/war/horror sites and magazines
  - High-quality subreddits (with a note on how story-dense they are)
  - Wikipedia category trees that spill out specific named stories
  - Digital archives / primary-source collections (records, trials, photos)
  - Newsletters, blogs, or podcast sites with story archives
  - YouTube channels whose transcripts teach us the proven-viral format

REQUIREMENTS:
- Concrete and checkable: give the actual URL/name. No vague "history blogs".
- Ranked by expected viral yield in OUR niche (e.g. a Unit 731 archive beats
  a generic military-history feed).
- Flag each source as: feed (rss), site (html), archive (api/json), wiki
  (category), reddit (subreddit), or channel (youtube).
- Skip: generic news wires, paywalled academia, low-quality aggregators.
- Aim for breadth across the three pillars — don't over-focus on one.

Respond with ONLY a JSON array, each object:
{{"name": "name", "url": "the feed/page/api endpoint URL", "kind": "feed|site|archive|wiki|reddit|channel", "pillar": "experiments|legends|ww1_ww2_stories", "rank": 1-10 (10 = highest viral yield), "note": "1 line on why it's valuable and how to fetch it"}}

Return the top {count} sources.
"""

GEMINI_TOPICS_PROMPT = """You brainstorm viral topic ideas for a faceless YouTube Shorts channel.

NICHE: CRAZY and brutal experiments run on HUMANS and ANIMALS by the great
powers in World War I and World War II (Unit 731, Nazi medical experiments,
chemical and radiation tests on prisoners and soldiers, secret labs that
treated living people as specimens), plus EPIC true stories from WW1/WW2
(impossible last stands, rescue missions, outnumbered units), and SCARY
legends, curses, hauntings and cover-ups rooted in real history. The money
angles are: SCARY (the horror and terror), INJUSTICE (victims betrayed or
used as pawns by their own governments, hidden truth, needless deaths),
HEROIC (outnumbered stands, impossible rescues), and MYSTERY (secret
programs, cover-ups of what really happened, unexplained events).

RULES:
- Each topic must be a CONCRETE named subject: a specific experiment, battle,
  operation, incident, unit, person, curse, legend or haunt. No vague
  concepts like "WW2".
- It must be REAL and DOCUMENTED (experiments, battles, history) — or a
  LEGEND/FOLKLORE story that is genuinely told and documented as such. Use
  your search grounding to verify each one actually has a record. Never
  invent or embellish.
- NEVER repeat any title in the BLOCKLIST below (exact or near-identical).
- Prefer topics that are little-known but well-documented — a dramatic story
  most people have never heard.
- No true crime, no serial killers, no missing-person cases, no boring
  weapon/aircraft spec catalogs.

Return ONLY a JSON array of {target} objects, each:
{{"title": "short topic title (max 12 words)", "angle": "mystery|scary|injustice|heroic", "summary": "1-2 sentence hook why this is fascinating"}}

BLOCKLIST (already used or too close to used topics — avoid all):
{blocklist}
"""


def resolve_topic_prompt(title: str, transcript: str) -> str:
    """Prompt that turns a competitor video transcript into a topic seed."""
    return (
        "You extract viral topic seeds from faceless documentary transcripts.\n"
        "Return ONLY a JSON object:\n"
        '{"topic": "concrete historical subject (named entity: person/operation/place/experiment), '
        'max 12 words", "angle": "mystery|scary|injustice|heroic", '
        '"niche_fit": "yes or no + 1-line reason"}\n'
        "Angle rules: mystery = secrets/cover-ups/unsolved unknowns; scary = horror/fear/terror "
        "in the story itself; injustice = institutional betrayal, innocent victims, cruel neglect; "
        "heroic = underdog soldiers, impossible stands, bravery. Never default to 'other'.\n"
        "Reject (niche_fit=no) if it's true crime, serial killers, missing persons, "
        "or boring weapon specs. Folklore and supernatural legends (ghosts, curses, "
        "hauntings, unexplained creatures, folklore) are ALLOWED when framed as a "
        "story or legend tied to documented history. Stick to WW1/WW2 stories, "
        "human/animal experiments, and dark legends.\n"
        f"Video title: {title}\nTranscript (first 6000 chars):\n{transcript[:6000]}"
    )