import json
import os
import re
import shutil

from src.config.settings import OUTPUT_DIR, TEMP_DIR


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{TEMP_DIR}/assets", exist_ok=True)
    os.makedirs(f"{TEMP_DIR}/audio", exist_ok=True)


def cleanup_temp():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def output_path(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    slug = slug[:60]  # cap length to avoid OS path issues
    return os.path.join(OUTPUT_DIR, f"{slug}.mp4")
