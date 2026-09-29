import hashlib
import json
import os
import re
import shutil
import threading
import time

DEBUG_DIR = "debug_output"

# Shared output/temp dirs (used directly by every service and agent that
# touches disk). Kept here as the single place that owns file paths.
OUTPUT_DIR = "output"
TEMP_DIR = "temp"

_ARTIFACT_SEQ = 0
_ARTIFACT_LOCK = threading.Lock()


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{TEMP_DIR}/assets", exist_ok=True)
    os.makedirs(f"{TEMP_DIR}/audio", exist_ok=True)


def workspace_path(topic: str) -> str:
    """Per-topic scratch dir so concurrent batch videos never collide.

    Each video's assets/audio/segments live under temp/<topic-slug> instead of
    a shared temp/ root. cleanup_temp() only ever removes a single workspace,
    so one finished video can't nuke a peer's in-flight files.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", (topic or "generic").lower()).strip("_")
    slug = slug[:40] or "video"
    return os.path.join(TEMP_DIR, slug)


def ensure_workspace(topic: str) -> str:
    ws = workspace_path(topic)
    os.makedirs(f"{ws}/assets", exist_ok=True)
    os.makedirs(f"{ws}/audio", exist_ok=True)
    return ws


def cleanup_temp(topic: str = None):
    target = workspace_path(topic) if topic else TEMP_DIR
    if os.path.exists(target):
        shutil.rmtree(target)


def dump_artifact(step: str, data, topic: str = "") -> str:
    """Write one pipeline step's output to debug_output/ for inspection.

    Strings are saved as .txt, everything else as .json. Each write gets a
    timestamp + sequence prefix so files never clobber each other within or
    across runs. Returns the file path.
    """
    global _ARTIFACT_SEQ
    with _ARTIFACT_LOCK:
        _ARTIFACT_SEQ += 1
        seq = _ARTIFACT_SEQ
    os.makedirs(DEBUG_DIR, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", (topic or "").lower()).strip("_")[:40]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{seq:02d}_{step}"
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


def output_path(topic: str) -> str:
    """Final MP4 path for a topic.

    The slug is capped at 60 chars, so two topics sharing a long prefix would
    otherwise write to the same file and silently overwrite each other. If the
    plain path is already taken, append a short hash of the FULL topic so both
    survive. A topic that has not been rendered yet keeps the exact path it
    always had, so existing output filenames are unchanged.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", (topic or "").lower()).strip("_")
    slug = slug[:60] or "video"  # cap length to avoid OS path issues
    path = os.path.join(OUTPUT_DIR, f"{slug}.mp4")
    if os.path.exists(path):
        digest = hashlib.sha1((topic or "").encode("utf-8")).hexdigest()[:6]
        path = os.path.join(OUTPUT_DIR, f"{slug}_{digest}.mp4")
    return path
