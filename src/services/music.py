"""Background-music mixing, applied after video assembly.

Kept out of video_assembler on purpose: assembly is a well-tested, surviving
module and mixing music into it would change narration output for every render.
Instead this is a post-assembly pass over the finished mp4, so
MUSIC_ENABLED=0 (or a missing/unreadable track) reproduces exactly the old
pipeline output.

How the mix is built
--------------------
* The track is read from ``MUSIC_SKIP_S`` (4s by default) because the opening
  sting is abrupt, then **looped** with ``-stream_loop -1`` so a video longer
  than the remaining audio does not go silent, and **trimmed** to the exact
  narration duration so the music never outlives the last word.
* Narration is padded to the same length first (``apad``) so ``amix`` never
  truncates the voice on a shorter music bed.
* TTS writes **mono 24 kHz** audio, and ffmpeg's ``aformat`` mono->stereo
  conversion normalizes by 1/sqrt(2) per channel, which quietly costs 3 dB of
  voice. The mono path therefore goes through ``pan`` instead, which centers
  the voice with no level change, and the choice is driven by the narration's
  real channel count so a stereo source is passed through untouched.
* ``normalize=0`` is essential: with ffmpeg's default normalization each input
  is divided by the input count, which would halve the narration too.
* The music bed is **loudness-normalised** (``loudnorm``) before the gain trim.
  A fixed dB offset alone is not a usable level control: it is relative to the
  track's own native loudness, so the same -18 dB that sounds right on a hot
  mastered track buries a quiet one to the point of silence. Normalising first
  makes ``MUSIC_TARGET_LUFS`` the actual level of the bed in the mix, and
  ``MUSIC_GAIN_DB`` a predictable trim on top of it.
* The sum of voice + music can exceed 0 dBFS, so a limiter catches the peaks
  rather than letting ffmpeg hard-clip them.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

from src.config.settings import (
    MUSIC_ENABLED,
    MUSIC_FADE_IN_S,
    MUSIC_FADE_OUT_S,
    MUSIC_GAIN_DB,
    MUSIC_PATH,
    MUSIC_SKIP_S,
    MUSIC_TARGET_LUFS,
)

NARRATION_RATE = 44100
NARRATION_LAYOUT = "stereo"


def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def probe_duration(path: str) -> float:
    """Duration of a media file in seconds via ffprobe.

    Tries the video stream first, then falls back to the container's format
    duration: a file whose video stream reports nothing (or is absent) still
    has a usable overall length.
    """
    for args in (
        ["-select_streams", "v:0", "-show_entries", "stream=duration"],
        ["-show_entries", "format=duration"],
    ):
        result = _run([
            "ffprobe", "-v", "error", *args,
            "-of", "json", path,
        ])
        if result.returncode != 0:
            continue
        try:
            data = json.loads(result.stdout or "{}")
        except ValueError:
            continue
        # ffprobe returns "streams" as a list but "format" as a single object.
        entries = data.get("streams")
        if not entries:
            fmt = data.get("format")
            entries = [fmt] if isinstance(fmt, dict) else (fmt or [])
        for entry in entries:
            try:
                value = float(entry.get("duration"))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    raise RuntimeError(f"could not determine duration of {path}")


def probe_audio_channels(path: str) -> int:
    """Channel count of the first audio stream (1 if undeterminable)."""
    result = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=channels", "-of", "json", path,
    ])
    try:
        data = json.loads(result.stdout or "{}")
        return int((data.get("streams") or [{}])[0].get("channels") or 1)
    except (ValueError, TypeError, IndexError):
        return 1


def build_filter(duration: float, gain_db: float = MUSIC_GAIN_DB,
                 fade_in_s: float = MUSIC_FADE_IN_S,
                 fade_out_s: float = MUSIC_FADE_OUT_S,
                 narr_channels: int = 1,
                 target_lufs: float = MUSIC_TARGET_LUFS) -> str:
    """The ffmpeg filter_complex that sums narration under the music bed.

    Split out from the ffmpeg call so the graph can be asserted directly in
    tests without spawning ffmpeg. ``narr_channels`` selects the upmix path:
    mono narration is centered with ``pan`` (level-preserving), anything else
    is already stereo and passes through. ``target_lufs`` is the loudness the
    bed is normalised to before ``gain_db`` trims it.
    """
    # Clamp so the fades can never overlap on a short clip.
    fade_in_s = min(fade_in_s, duration / 2)
    fade_out_s = min(fade_out_s, max(0.0, duration - fade_in_s))
    fade_out_start = max(0.0, duration - fade_out_s)
    # Level-preserving mono -> stereo: pan duplicates c0 into both channels,
    # whereas aformat's channel conversion would divide by sqrt(2) per channel.
    upmix = "pan=stereo|c0=c0|c1=c0," if narr_channels == 1 else ""
    return (
        # Pad the narration to the full length so the voice is never truncated
        # by a shorter music bed, then pin its timestamps.
        f"[0:a]aformat=sample_rates={NARRATION_RATE},"
        f"{upmix}"
        f"apad,atrim=0:{duration:.3f},asetpts=N/SR/TB[n];"
        # Trim/loop-free bed at the same rate, loudness-normalised to a known
        # target so the mix level does not depend on the track's own loudness,
        # then trimmed and softened at the edges.
        f"[1:a]aformat=sample_rates={NARRATION_RATE}:channel_layouts={NARRATION_LAYOUT},"
        f"loudnorm=I={target_lufs:.1f}:TP=-2.0:LRA=11.0,"
        f"atrim=0:{duration:.3f},asetpts=N/SR/TB,"
        f"afade=t=in:st=0:d={fade_in_s:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_s:.3f},"
        f"volume={gain_db:.1f}dB[m];"
        # normalize=0 keeps the narration at its own level; the limiter catches
        # the few samples where voice + music sum past 0 dBFS.
        f"[n][m]amix=inputs=2:duration=longest:normalize=0,"
        f"alimiter=limit=0.95:level=disabled[aout]"
    )


def music_available(path: str = MUSIC_PATH) -> bool:
    """True when music mixing is enabled and the track is actually usable."""
    if not MUSIC_ENABLED:
        return False
    if not path or not os.path.isfile(path):
        return False
    if shutil.which("ffmpeg") is None:
        return False
    return _run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=codec_type", "-of", "csv=p=0", path]).returncode == 0


def add_background_music(video_path: str, out_path: str = None,
                         music_path: str = MUSIC_PATH,
                         gain_db: float = MUSIC_GAIN_DB) -> str:
    """Mix the background track under the narration of ``video_path``.

    Returns the path of the final video. Returns ``video_path`` unchanged when
    music is disabled or unavailable, so callers can treat this as a no-op
    filter in the pipeline. The mixed file is written next to the input (or to
    ``out_path``) and the input is left on disk.
    """
    if out_path is None:
        out_path = os.path.splitext(video_path)[0] + "_music.mp4"

    if not music_available(music_path):
        reason = "disabled" if not MUSIC_ENABLED else f"unavailable ({music_path})"
        print(f"  [music] Skipping background music: {reason}")
        return video_path

    duration = probe_duration(video_path)
    channels = probe_audio_channels(video_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        # -ss before -i on the audio input seeks fast; -stream_loop -1 keeps
        # supplying music past the end of the track.
        "-ss", f"{MUSIC_SKIP_S}",
        "-stream_loop", "-1",
        "-i", music_path,
        "-filter_complex", build_filter(duration, gain_db, MUSIC_FADE_IN_S, MUSIC_FADE_OUT_S,
                         channels),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",                 # never re-encode the rendered video
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{duration:.3f}",
        "-movflags", "+faststart",
        out_path,
    ]
    result = _run(cmd)
    if result.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError(
            f"music mix failed (exit {result.returncode}): {result.stderr[-600:]}"
        )

    print(f"  [music] Mixed '{os.path.basename(music_path)}' at {gain_db:.1f} dB "
          f"under narration ({duration:.1f}s, {channels}ch) → {out_path}")
    return out_path
