"""Research agent: sources fresh niche material -> structured video ideas.

The deep path (``run_research_pipeline``) pulls Wikipedia categories + search
per niche, runs the local model (Ollama, via ``src.agents.common.llm``) over
source chunks to extract structured video ideas, dedups against everything
already produced, scores for novelty/angle-balance, and writes an approved
queue to topics/batch_*.json — the same queue the video pipeline reads.

The light path (``generate_first_topics``) does zero scraping: one cheap LLM
call per niche proposes titles, we take the FIRST `target` titles we have not
already made a video about, and produce.

``sources.py`` holds the per-topic researcher used at script time; prompts,
parsing, and the niche map live in ``prompts.py`` / ``helpers.py``.
"""