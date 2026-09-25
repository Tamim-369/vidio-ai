"""Prompt that discovers the best sources for viral topics in our niche.

Given the niche below, the model lists concrete, usable content sources
(feeds, sites, subreddits, Wikipedia categories, archives, etc.) ranked by
how reliably they produce viral-ready stories. Output is structured JSON so a
future step can turn it straight into the topic-agent source inventory
(src/services/topic_agent/sources.py).
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