"""The anti-wisdom quote prompt.

QUOTE_PROMPT is the user's specification, reproduced verbatim: the generation
rules, format rotation and self-check list are the contract, and the agent
re-sends rejected quotes with feedback, so wording changes alter output quality.

Two things live outside the prompt text and are applied by the agent:
  * OUTPUT_FORMAT asks for one JSON object per line so a batch parses cleanly.
    The agent strips labels itself rather than trusting the model to comply.
  * The already-used quotes and figures are supplied each round, so rotation is
    enforced across videos and not just within one reply."""

# The user's prompt, verbatim.
QUOTE_PROMPT = '''Generate short, funny "anti-wisdom" quotes.

Rules:

No real wisdom. Punchline can't be a valid point or clever logic, even ironic. If it makes actual sense, it's wrong.
One punchline, one beat. No "and then," no stacked scenarios, no mini-stories.
Twist must stay on the same topic as the setup — don't swap to something unrelated.
Plain, spoken words only. No essay vocabulary. No meme-crutch phrases.
Default short — one sentence + one punch. Only go longer for one sharp concrete detail, never a scene.
When twisting a real quote/proverb, you MUST change the wording of the ending/key phrase into something dumb. Never just pair it with a second real proverb — that's still real wisdom, just doubled.
Source pool for "twisted proverb" quotes: pull specifically from well-known historical wisdom figures — Sun Tzu, Greek philosophers (Socrates, Plato, Aristotle, Epictetus, etc.), Confucius, and classic proverbs. Rotate widely across this pool, don't reuse the same figure/quote repeatedly.
When using a fake attribution, the quote itself must also be altered/made-up — never attach a fake name to an untouched real quote.
No recycled meme lines — don't reuse existing internet jokes/t-shirt slogans. Generate something new.
Rotate formats across a batch: twisted proverb (from the source pool above), fake attribution + altered quote, original one-liner, crude/innuendo.
Output only the final quotes — no format labels, no self-correction, no meta-commentary. If a quote turns out to be real/unaltered, silently discard and regenerate instead of narrating it.

Self-check before finalizing each quote:

Does it secretly make sense / is it actually a fair point? → fix.
Is it just two real sayings paired together? → fix.
Is the fake-attributed quote actually altered, not just relabeled? → fix.
Is this a recycled meme I've seen before? → discard, make a new one.
Any leftover labels, notes, or self-talk in the output? → strip it out.
Did I pull from Sun Tzu / Greek philosophers / similar classic sources for the twisted-proverb format? → check rotation.'''

# Appended per request so a batch parses deterministically. The agent also
# enforces "no labels, no meta-commentary" itself in quote_agent._clean_candidate.
OUTPUT_FORMAT = '''

Return {n} quotes as a JSON array of objects, nothing else:
[{{"quote": "<the quote text>", "format": "<twisted_proverb|fake_attribution|one_liner|crude>", "source": "<the figure or proverb it twists, or empty>"}}]

Every quote must be a single beat with a dumb punchline. No labels, no commentary, no markdown fences.'''

# Historical wisdom figures named in the prompt. Used for rotation tracking so
# the same figure is not twisted twice in a row.
SOURCE_FIGURES = (
    "sun tzu",
    "socrates",
    "plato",
    "aristotle",
    "epictetus",
    "confucius",
    "lao tzu",
    "plutarch",
    "thales",
    "heraclitus",
    "diogenes",
    "seneca",
    "marcus aurelius",
    "buddha",
)
