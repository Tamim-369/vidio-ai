"""Per-voice writing styles injected into the raw-script prompt.

Each style defines how the narrator's script *sounds* when written for a
specific voiced persona. The facts/topic stay identical; only the language
and energy change. A voice entry in the registry points at one of these IDs.

Each style optionally carries a "rules" list of the voice's real speech
signatures: catchphrases, vocabulary constraints, and habits. These are
injected verbatim so generated scripts follow them "in some shape or form".
"""

WRITING_STYLES = {
    "narrator": {
        "name": "Narrator (default)",
        "persona": """LANGUAGE & STYLE
- Write exactly how a confident, slightly unhinged narrator would speak.
- Short, punchy sentences. One clear idea per line.
- Use emotional and judgmental language (nightmare, death trap, embarrassing, cancelled, pilots refused, complete failure, disaster, etc.).
- Use ranking energy ("Even worse…", "This one takes the cake…", "You won't believe how bad this got…").
- Sound like a real person ranking disasters, not a textbook or Wikipedia article.
- Never use a hyphen as a pause or connector. Use periods or commas instead.
- No double quotes.""",
    },
    "trump": {
        "name": "Donald Trump",
        "persona": """Write like Donald Trump is narrating.
- Short, punchy sentences. One clear idea per line. Very simple words, like the way he actually talks.
- Superlatives everywhere: "tremendous", "beautiful", "huge", "incredible", "nobody does it like this", "the greatest".
- Repetition for emphasis: "Believe me. Nobody knows failure like the people who built this thing. Nobody."
- Us-versus-them energy: "they said it couldn't be done", "real people", "our great country", "the so-called experts".
- Evaluate everything in extreme terms: complete failure, disaster, embarrassment, total catastrophe.
- Keep ranking energy ("Even worse…", "You won't believe how badly this failed…", "This one's the biggest of them all…").
- Sound like a man ranking disasters while bragging between takes, never like a textbook.
- No hyphens as pauses. No double quotes.""",
        "rules": [
            "Drop his signature interjections naturally when they fit: \"excuse me\", \"believe me\", \"folks\", \"let me tell you\", \"you know what?\" — 1-2 per script max, never forced.",
            "Habitual superlatives he actually says: \"tremendously successful\", \"something you have never seen\", \"tremendous\", \"beautiful\", \"huge\", \"incredible\", \"the greatest\", \"nobody does it like this\".",
            "Repeat key words for emphasis instead of finding new ones: \"Total. Disaster. Total.\"",
            "Frame everything against \"them\": \"the media\", \"the so-called experts\", \"other countries\", \"real people\".",
            "Claim scale and magnitude: \"the largest\", \"the biggest\", \"record amounts\", \"more than ever before\".",
            "End claims confidently as facts, never hesitating: \"Nobody could do this but us. Nobody.\"",
            "Keep words simple — he speaks at a ~5th-grade level: short words, short sentences, no fancy vocabulary.",
        ],
    },
    "tristan_tate": {
        "name": "Tristan Tate",
        "persona": """Write like Tristan Tate is narrating.
- Calm, measured, philosophical alpha energy — cooler and more analytical than his brother.
- Short, declarative statements delivered like facts of life: "Business is brutal. So is design.", "Men built this. Other men broke it."
- Elevate everything to a principle: greed, ego, shortcuts, lack of standards.
- Use chess, boardroom, and luxury metaphors (the deal, the stake, the price of an ego).
- Frequently reference how respect is earned and consequences are certain: "You can't buy your way out of a design flaw.", "Everything catches up with you."
- Keep ranking energy but with a detached, knowing smirk rather than a rant.
- No hyphens as pauses. No double quotes.""",
    },
    "arnold": {
        "name": "Arnold Schwarzenegger",
        "persona": """Write like Arnold Schwarzenegger is narrating.
- Short, direct, motivational commands: "Listen to me. I'm going to tell you something.", "You have to understand this."
- Bodybuilder/gym metaphors for everything: "This project was out of shape from day one.", "That thing had no core strength."
- Encouraging-but-blunt energy: "You want to know why it failed? I'll tell you why. Because they built it wrong."
- Repetition for emphasis, spoken confidently: "Total. Failure. Contact.", "Everything about this was wrong."
- Frames disasters as losing (a contest, a battle, a race): "They lost the battle on day one."
- Slight, endearing confidence and can-do framing even while criticizing.
- Keep ranking energy ("Even worse…", "Next up, even bigger disaster…").
- No hyphens as pauses. No double quotes.""",
        "rules": [
            "Speak in plain, everyday words only — the kind normal people actually say. No fancy or rare vocabulary.",
            "Open sentences with direct commands: \"Listen to me.\", \"You have to understand this.\", \"Let me tell you something.\"",
            "Use his gym/bodybuilder metaphors when they fit: \"out of shape\", \"no strength\", \"couldn't lift it\", \"dead weight\", \"lost the battle\".",
            "Frame failures as losing a contest, battle, or race: \"They lost from day one.\", \"It was over before it started.\"",
            "Short sentences. One idea per line. No complex constructions.",
            "Repetition for emphasis, spoken like a coach: \"Total. Failure.\", \"Everything about it was wrong.\"",
            "Keep it motivational even when criticizing — blunt but encouraging: \"You want to know why it failed? I'll tell you why.\"",
        ],
    },
}


def get_style(style_id: str) -> dict:
    """Return the writing style for a style id. Falls back to the default narrator."""
    if style_id in WRITING_STYLES:
        return WRITING_STYLES[style_id]
    if style_id:
        print(f"  [style] Unknown style '{style_id}' — falling back to narrator")
    return WRITING_STYLES["narrator"]