"""Runtime knobs for the staged script pipeline (env-overridable).

All tuning values for the local-lab path live here so they are discoverable
in one place: which model writes, how long any stage may run before it is
declared wedged, and how many tokens each stage is allowed to generate.
"""
import os

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen2.5:3b")
STAGE_TIMEOUT_S = int(os.getenv("STAGE_TIMEOUT_S", "300"))
NUM_PREDICT = {
    "angle": 500,
    "facts": 800,
    "story": 900,
    "theme": 350,
    "json": 1600,
    "query": 2000,
    "research": 2000,
    "kw": 500,          # asset_agent: topic -> keyword phrases
    "assign": 1200,     # asset_agent: image -> line JSON mapping
    "as_repair": 800,   # asset_agent: one self-correction pass
    "queries": 3500,    # asset_agent: per-line search queries JSON
}