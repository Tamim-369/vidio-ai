"""Run-scoped directories and debug artifact dumps for a video run."""

import json
import os
import re
import shutil
import time

OUTPUT_DIR = "output"
TEMP_DIR = "temp"
DEBUG_DIR = "debug_output"

_ARTIFACT_SEQ = 0


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{TEMP_DIR}/assets", exist_ok=True)
    os.makedirs(f"{TEMP_DIR}/audio", exist_ok=True)


def cleanup_temp():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


def dump_artifact(step: str, data, topic: str = "") -> str:
    """Write one pipeline step's output to debug_output/ for inspection.

    Strings are saved as .txt, everything else as .json. Each write gets a
    timestamp + sequence prefix so files never clobber each other within or
    across runs. Returns the file path.
    """
    global _ARTIFACT_SEQ
    _ARTIFACT_SEQ += 1
    os.makedirs(DEBUG_DIR, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", (topic or "").lower()).strip("_")[:40]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{_ARTIFACT_SEQ:02d}_{step}"
    if slug:
        name += f"_{slug}"
    suffix = ".txt" if isinstance(data, str) else ".json"
    path = os.path.join(DEBUG_DIR, name + suffix)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [artifact] {step} -> {path}")
    return path
