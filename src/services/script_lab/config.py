"""Runtime knobs for the staged script pipeline (env-overridable).

All tuning values for the local-lab path live here so they are discoverable
in one place: which model writes, how long any stage may run before it is
declared wedged, and how many tokens each stage is allowed to generate.
"""
import os

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "phi4-mini:latest")
STAGE_TIMEOUT_S = int(os.getenv("STAGE_TIMEOUT_S", "300"))
NUM_PREDICT = {
    "angle": 500,
    "facts": 800,
    "story": 900,
    "theme": 350,
    "json": 1600,
    "query": 2000,
}