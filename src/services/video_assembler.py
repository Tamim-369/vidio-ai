"""Joining rendered quote-card segments into one video.

Only the concat step survives from the old assembler. `assemble()` used to turn
a researched script plus stock images into a Ken Burns slideshow with burned-in
captions; the quote card renders each line as a single still whose duration is
the narration length, so there are no frames to zoom, crossfade or caption here.

What remains is the ffmpeg concat demuxer step that joins the per-quote cards
into a single file. quote_card.render() calls it once at the end.
"""
from __future__ import annotations

import os
import subprocess

from src.config.settings import TEMP_DIR


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
