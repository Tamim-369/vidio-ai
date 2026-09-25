"""Agent packages: each self-contained unit owns its prompts, helpers and
supporting modules, and is driven by src/pipeline/* as orchestrator.

    common/    shared deterministic plumbing (config, llm, text)
    scene/     pick one incident from material -> scene brief
    story/     writer + showrunner + fact-gate -> story draft
    script/    story -> spoken scene lines
    query/     per-line / per-image search queries
    topic/     topic sourcing (reddit/wikipedia/gemini/channel mining)
    research/  research pipeline for the daily topic round
    asset/     image gathering + line->image assignment
    voice/     voice selection and writing-style lookup
"""