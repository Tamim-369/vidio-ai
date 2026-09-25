"""Per-voice writing styles used to select how a voiced script is written.

Each style defines how the narrator's script *sounds* when written for a
specific voiced persona. The facts/topic stay identical; only the language
and energy change. A voice entry in the registry points at one of these IDs.

The main branch has a single voice (the Pocket-TTS narrator), so only the
default narrator style ships here. Persona styles (chatterbox clones) live
on the wizdom branch.

Each style optionally carries a "rules" list of the voice's real speech
signatures: catchphrases, vocabulary constraints, and habits. These are
injected verbatim so generated scripts follow them "in some shape or form".
"""

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
}


def get_style(style_id: str) -> dict:
    """Return the writing style for a style id. Falls back to the default narrator."""
    if style_id in WRITING_STYLES:
        return WRITING_STYLES[style_id]
    if style_id:
        print(f"  [style] Unknown style '{style_id}' — falling back to narrator")
    return WRITING_STYLES["narrator"]