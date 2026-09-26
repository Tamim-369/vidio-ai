"""Joins the rendered quote-card segments into one file (ffmpeg concat, no re-encode)."""
from __future__ import annotations

import os
import subprocess

TEMP_DIR = "temp"


def concat_segments(segment_paths: list, out: str) -> None:
    """Concatenate pre-rendered mp4 segments with ffmpeg concat demuxer (no re-encode)."""
    list_file = os.path.join(TEMP_DIR, "concat_list.txt")
    with open(list_file, "w") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
