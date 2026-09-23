"""Local asset agents — keyword, query, and image-assignment passes.

Replace the Groq/Gemini calls that used to drive asset fetching with a single
local seam: Ollama (phi4-mini via the staged lab's ``_local``) first, a
deterministic heuristic fallback second. No cloud provider is ever called from
the asset stage, so image selection cannot stall on Groq key rot / Gemini 503s.

Each agent returns a usable result on failure (keyword/query/assignment all
have a non-LLM path), so asset fetching never blocks on the model.
"""