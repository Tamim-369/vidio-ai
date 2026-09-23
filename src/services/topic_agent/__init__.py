"""Topic agent: sourced topic leads -> normalized -> deduped -> scored -> batch.

Replaces the blank-prompt topic generation with collector-driven discovery:
each source emits leads (title + url + snippet + optional long-form body),
which the agent dedups against everything already made, scores with a
virality heuristic, and selects into the same topics/batch_*.json queue the
video pipeline already reads. Everything here is web scraping only — no
Groq/Gemini, no Reddit.
"""