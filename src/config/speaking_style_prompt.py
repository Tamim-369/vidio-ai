"""Prompt that discovers what is needed to BUILD a speaking style for a narrator.

The story layer of the script pipeline writes the entire narration in a
character's voice in ONE pass. Before we can build those speaking styles we
need to know what a speaking style must contain. This prompt asks another AI
to tell us exactly that — it does NOT prescribe the answer.
"""

SPEAKING_STYLE_PROMPT = """You are advising me on how to build "speaking styles" — profiles that let an AI narrator replicate how a real person actually talks, so the AI can retell true stories exactly as if that person were telling them.

I am NOT going to tell you what a speaking style should contain. That is your job. I need you to tell me what we need.

CONTEXT
- A small local AI model will use the speaking style to write a full spoken story in that person's voice, in one pass, from a factual source. The facts and numbers stay exact — only the voice changes.
- I want the AI to sound like the actual person, not a parody built from a few catchphrases.

FOR EACH OF THESE THREE PEOPLE:
1. Arnold Schwarzenegger
2. Donald Trump
3. Andrew Tate

TELL ME:
- What are the essential ingredients of a speaking style that would make an AI sound like this person? Think in depth: speech patterns, vocabulary, sentence shape and rhythm, catchphrases and verbal tics, tone and energy, worldview and opinions, how they rank and judge things, what metaphors and references they reach for, how they open and how they close.
- For every ingredient you list, explain why it matters and how an AI could actually use it while writing.
- What is the minimum set of ingredients without which the persona falls apart?
- What are the common mistakes people make when replicating this person's voice, and how should the profile guard against them?
- Given neutral factual material (real historical stories), how would this person's voice bend the storytelling WITHOUT changing the facts?

Be concrete and specific, not vague. Structure your answer so each person's speaking style can be built directly from it.
"""