"""Topic agent: sourced topic leads -> normalized -> deduped -> scored -> batch.

Two generation paths:
  * ``agent.py`` / ``run_topic_agent`` — collector-driven discovery: each
    source emits leads (title + url + snippet + optional long-form body),
    which the agent dedups against everything already made, scores with a
    virality heuristic, and selects into the same topics/batch_*.json queue the
    video pipeline already reads. Web scraping only — no inferred topics.
  * ``generator.py`` / ``run_topic_generation`` — blank-prompt generation
    (LLM brainstorm + channel mining + Wikipedia + Reddit backstop).

Shared deterministic helpers (dedupe, used/made tracking, niche filters, used
topic queue) live in ``helpers.py``; the Wikipedia crawl in ``wikipedia.py``;
prompts in ``prompts.py``. ``http.py``/``collectors.py``/``leads.py``/
``sources.py``/``score.py`` are the source-layer machinery for the collector
path.
"""