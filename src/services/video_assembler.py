import os
import numpy as np
from PIL import Image
from moviepy import VideoClip, AudioFileClip, concatenate_videoclips
from src.config.settings import VIDEO_FORMAT, VIDEO_RESOLUTIONS
from src.utils.file_helpers import output_path

FPS = 24


def _load_image(img_path: str, size: tuple) -> np.ndarray:
    """Load, resize and center-crop image to exact target size."""
    w, h = size
    img = Image.open(img_path).convert("RGB")

    img_ratio = img.width / img.height
    target_ratio = w / h

    if img_ratio > target_ratio:
        new_h, new_w = h, int(img.width * h / img.height)
    else:
        new_w, new_h = w, int(img.height * w / img.width)

    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return np.array(img.crop((left, top, left + w, top + h)))


def _make_zoom_frames(img_array: np.ndarray, duration: float, size: tuple) -> np.ndarray:
    """Pre-render all frames with Ken Burns zoom. Returns (n_frames, H, W, 3) array."""
    w, h = size
    n_frames = max(1, int(duration * FPS))
    zoom_start, zoom_end = 1.0, 1.08
    pil_img = Image.fromarray(img_array)
    frames = []

    for i in range(n_frames):
        scale = zoom_start + (zoom_end - zoom_start) * (i / max(n_frames - 1, 1))
        new_w, new_h = int(w * scale), int(h * scale)
        img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        frames.append(np.array(img.crop((left, top, left + w, top + h))))

    return np.stack(frames)  # shape: (n_frames, H, W, 3)


def assemble(script: dict) -> str:
    size = VIDEO_RESOLUTIONS[VIDEO_FORMAT]
    clips = []

    for line in script["lines"]:
        if not line.get("asset_path") or not line.get("audio_path"):
            print(f"  [assembler] Skipping line {line['id']} — missing asset or audio")
            continue

        duration = line["actual_duration"]

        print(f"  [assembler] Pre-rendering zoom frames for line {line['id']}...")
        frames = _make_zoom_frames(_load_image(line["asset_path"], size), duration, size)

        # VideoClip with frame lookup into pre-rendered array — fast
        def make_frame(t, f=frames):
            idx = min(int(t * FPS), len(f) - 1)
            return f[idx]

        video_clip = VideoClip(make_frame, duration=duration)
        audio = AudioFileClip(line["audio_path"])
        video_clip = video_clip.with_audio(audio)

        clips.append(video_clip)

    final = concatenate_videoclips(clips, method="compose")
    out = output_path(script["topic"])

    print(f"  [assembler] Rendering → {out}")
    final.write_videofile(out, fps=FPS, codec="libx264", audio_codec="aac", logger=None)

    return out
