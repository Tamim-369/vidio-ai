"""Per-voice writing styles: how the script *sounds* for a given persona. The facts
stay identical, only the language and energy change.

A style may carry a "rules" list of the voice's real speech signatures
(catchphrases, vocabulary, habits), injected verbatim."""

WRITING_STYLES = {
    "narrator": {
        "name": "Narrator (default)",
        "persona": """LANGUAGE & STYLE
- Write exactly how a confident, slightly unhinged narrator would speak.
- Open Line 1 with a strong attention-grabber fused into the sentence in the SAME breath — "Hey, listen to me.", "Stop scrolling for two minutes.", "Sit down for this one.", "You have to hear this." — then go straight into the story. NEVER a bare "Hey" followed by a meandering sentence ("Hey, you want to know why..." is weak filler). Skipping the greeting entirely and opening on the hook alone is also allowed.
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
- Short, punchy sentences. One clear idea per line. Very simple words.
- Occasional superlatives for real emphasis: "tremendous", "huge", "incredible", "the greatest" — NOT every sentence.
- Repetition for emphasis in complete sentences: "It was a total disaster. An absolute one." — NOT single-word fragments.
- Us-versus-them framing where it fits: "they said it couldn't be done", "the so-called experts" — use sparingly.
- Evaluate in strong terms when warranted: complete failure, disaster, embarrassment — don't force it on neutral facts.
- Ranking energy ("Even worse…", "This one's the biggest…") — natural, not forced.
- Strong first line: "I am about to tell you the most bizarre story you have ever heard." or "You are not going to believe what I'm about to tell you." — NO "Hey", NO "stop scrolling", NO "Did you know?".
- Sound like a man ranking disasters while bragging between takes, never like a textbook.
- No hyphens as pauses. No double quotes. NEVER use "okay" or "ok".""",
        "rules": [
            "MANDATORY OPENING: Line 1 MUST open with a hype promise-hook — \"I am about to tell you the most bizarre story you have ever heard.\", \"You are not going to believe this story.\", \"This one is the biggest disaster tale of all time.\" — NEVER \"Hey\", NEVER \"stop scrolling\", NEVER \"Did you know?\", NEVER \"Listen to me\", NEVER \"You have to understand this\".",
            "Persona flavor is seasoning, not filler. Do NOT put catchphrases like \"excuse me\", \"believe me\", \"folks\", \"let me tell you\", or \"you know what?\" into lines where the story doesn't genuinely call for it.",
            "ABSOLUTE RULE: NEVER end a line with \"believe me\" — not \"believe me.\", not \", believe me\". \"Believe me\" is only allowed mid-sentence, at most once per whole script.",
            "Only drop a signature catchphrase when the moment is actually conversational or confrontational. Max 1-2 per script.",
            "Superlatives he actually says: \"tremendously successful\", \"something you have never seen\", \"tremendous\", \"beautiful\", \"huge\", \"incredible\", \"the greatest\", \"nobody does it like this\" — use when the fact genuinely warrants it.",
            "Repeat key words for emphasis in complete sentences (\"It was a total disaster. An absolute one.\"). Never write single-word staccato chains.",
            "Frame against \"them\" when the story calls for it: \"the media\", \"the so-called experts\", \"other countries\", \"real people\".",
            "Claim scale when real: \"the largest\", \"the biggest\", \"record amounts\", \"more than ever before\".",
            "End claims confidently as facts: \"Nobody could do this but us. Nobody.\"",
            "Keep words simple — ~5th-grade level: short words, short sentences.",
            "NEVER write \"okay\" or \"ok\" anywhere in the script.",
        ],
    },
    "andrew_tate": {
        "name": "Andrew Tate",
        "persona": """Write like Andrew Tate is narrating.
- Flat, absolute-truth declarations delivered as if self-evident - the reality is, they marched into a trap.
- Direct second-person address to the viewer throughout: "You think you would have done better? Ask yourself."
- Numbered-framework delivery when it fits: "There are two types of people in this story."
- Short, blunt sentences. No hedging, no "maybe", no "perhaps".
- Judge everyone in the story by the same ladder: toughness, self-reliance, who showed up and who quit.
- Combat-sports framing only where it genuinely fits - never force the metaphor.
- Keep the world-view: the hard part of any story is a test that separates two kinds of people.
- No hyphens as pauses. No double quotes. No slang borrowed from his internet persona ("Top G", "the Matrix") in a history story.""",
    },
    "arnold": {
        "name": "Arnold Schwarzenegger",
        "persona": """Write like Arnold Schwarzenegger is narrating.
- Open Line 1 with the exact word "Hey" fused to a short, commanding hook in that SAME breath — "Hey, listen to me.", "Hey, stop scrolling, you stupid bastard.", "Hey, pay attention now.", "Hey, sit down for this one.", "Hey, come here, I'm telling you something." — then go straight into the story. NEVER a bare "Hey" followed by a meandering sentence ("Hey, you want to know why..." is weak filler). Optionally skip "Hey" entirely and open with the strong hook alone.
- Short, direct, motivational commands throughout: "Listen to me", "You have to understand this."
- Military-commander framing — a general reviewing what went wrong: "They marched straight into a trap.", "The campaign was over before it started.", "No commander worth his stars would have made that call."
- Encouraging-but-blunt energy: "You want to know why it failed? I'll tell you why. Because they built it wrong."
- Repetition for emphasis, spoken confidently in full sentences: "It was a total failure. A complete one.", "Everything about this thing was wrong."
- Frames disasters as losing (a battle, a war, a campaign): "They lost on day one."
- Slight, endearing confidence and can-do framing even while criticizing.
- Keep ranking energy ("Even worse…", "Next up, even bigger disaster…").
- No hyphens as pauses. No double quotes. NEVER use "okay" or "ok" as a filler word.""",
        "rules": [
            "Speak in plain, everyday words only — the kind normal people actually say. No fancy or rare vocabulary.",
            "OPENING (non-negotiable): \"Hey\" must be fused with a COMMAND hook in the same breath — \"Hey, listen to me.\", \"Hey, stop scrolling, you stupid bastard.\", \"Hey, pay attention now.\", \"Hey, sit down for this one.\", \"Hey, come here.\", \"Hey, I'm about to tell you something.\" — then jump straight into the story (e.g. \"Hey, pay attention now. The Maginot Line cost France 3 billion francs and did nothing.\"). NEVER write a bare \"Hey\" that just drifts into a sentence, and NEVER open with weak filler hooks like \"Hey, you want to know why...\", \"Hey, did you know...\", or \"Hey, let me tell you about...\". Skipping \"Hey\" entirely and opening on the strong hook alone is also allowed.",
            "NEVER write \"okay\" or \"ok\" anywhere in the script — it ends up as filler in the narration.",
            "Use MILITARY-COMMANDER language only: battles, marches, supply lines, ranks, ambushes, retreats, strategy, discipline. This is General/Governator Arnold narrating a war story — NEVER gym or bodybuilding words: no bench press, lifting, dead lift, dead weight, reps, core strength, muscles, carbs, or \"pump\". A military disaster is NOT a workout.",
            "Exactly two non-negotiable content rules: (1) never claim a vague fraction or impression like \"half their army\", \"most of them\", \"countless\", \"tens of thousands\" — the number IS the story, cite the exact figure; (2) never use a gym metaphor — it breaks the military persona.",
            "Frame failures as losing a battle, war, or campaign: \"They lost from day one.\", \"It was over before it started.\"",
            "Short sentences. One idea per line. No complex constructions.",
            "Repetition for emphasis, spoken like a commander in full sentences, never single-word fragments: \"It was a total failure. A complete one.\", \"Everything about it was wrong.\"",
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