import os
import subprocess
import numpy as np
from PIL import Image
from moviepy import VideoClip, AudioFileClip
from src.config.settings import VIDEO_FORMAT, VIDEO_RESOLUTIONS, TEMP_DIR
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
    zoom_start, zoom_end = 1.0, 1.06  # Much more subtle zoom (was 1.12)
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


def _crossfade_frames(frames1: np.ndarray, frames2: np.ndarray, fade_frames: int) -> np.ndarray:
    """Crossfade between two frame arrays. Returns combined array."""
    if fade_frames <= 0:
        return np.concatenate([frames1, frames2], axis=0)
    
    # Take last fade_frames from frames1 and first fade_frames from frames2
    fade_out = frames1[-fade_frames:]
    fade_in = frames2[:fade_frames]
    
    # Create alpha blend
    faded = []
    for i in range(fade_frames):
        alpha = i / fade_frames  # 0.0 to 1.0
        blended = (fade_out[i] * (1 - alpha) + fade_in[i] * alpha).astype(np.uint8)
        faded.append(blended)
    
    # Combine: frames1 (except last fade_frames) + faded + frames2 (except first fade_frames)
    return np.concatenate([
        frames1[:-fade_frames] if len(frames1) > fade_frames else frames1,
        np.stack(faded),
        frames2[fade_frames:] if len(frames2) > fade_frames else frames2
    ], axis=0)


def _line_frames(asset_paths: list, total_duration: float, size: tuple) -> np.ndarray:
    """Build the frame array for a single line, bounded to that line only."""
    CROSSFADE_DURATION = 0.8  # Slower crossfade: 800ms instead of 300ms
    MIN_IMAGE_DURATION = 2.5  # Minimum time per image: 2.5 seconds

    num_images = len(asset_paths)

    if num_images == 1:
        return _make_zoom_frames(_load_image(asset_paths[0], size), total_duration, size)

    # Multiple images: ensure minimum duration per image
    raw_time_per_image = total_duration / num_images
    original_count = num_images

    if raw_time_per_image < MIN_IMAGE_DURATION:
        # Use fewer images to meet minimum duration
        max_images = int(total_duration / MIN_IMAGE_DURATION)
        if max_images > 0:
            asset_paths = asset_paths[:max_images]
            num_images = len(asset_paths)
            time_per_image = total_duration / num_images
            print(f"  [assembler] Using {num_images} images (reduced from {original_count}) for proper timing")
        else:
            # Fallback: use single image if duration is very short
            asset_paths = [asset_paths[0]]
            num_images = 1
            time_per_image = total_duration
    else:
        time_per_image = raw_time_per_image

    fade_frames = int(CROSSFADE_DURATION * FPS)
    max_fade_frames = int((time_per_image * 0.3) * FPS)  # Max 30% of image time
    fade_frames = min(fade_frames, max_fade_frames)

    print(f"  [assembler] Each image: {time_per_image:.1f}s, crossfade: {fade_frames/FPS:.1f}s")

    all_frames = []
    for i, img_path in enumerate(asset_paths):
        print(f"  [assembler] Pre-rendering image {i+1}/{num_images}...")
        img_array = _load_image(img_path, size)
        img_frames = _make_zoom_frames(img_array, time_per_image, size)

        if i == 0:
            all_frames = img_frames
        else:
            all_frames = _crossfade_frames(all_frames, img_frames, fade_frames)

    return all_frames


def _concat_segments(segment_paths: list, out: str) -> None:
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


def assemble(script: dict) -> str:
    size = VIDEO_RESOLUTIONS[VIDEO_FORMAT]
    os.makedirs(TEMP_DIR, exist_ok=True)
    segment_paths = []
    line_duration = 0.0

    for line in script["lines"]:
        audio_path = line.get("audio_path")
        asset_paths = line.get("asset_paths", [])
        
        # Backward compatibility: check for single asset_path
        if not asset_paths and line.get("asset_path"):
            asset_paths = [line.get("asset_path")]
        
        if not asset_paths or not audio_path:
            print(f"  [assembler] Skipping line {line['id']} — missing assets or audio")
            continue

        total_duration = line["actual_duration"]

        print(f"  [assembler] Line {line['id']}: Processing {len(asset_paths)} image(s) over {total_duration:.1f}s...")

        # Build ONLY this line's frames, then render it to a temp file and free.
        frames = _line_frames(asset_paths, total_duration, size)

        def make_frame(t, f=frames):
            idx = min(int(t * FPS), len(f) - 1)
            return f[idx]

        video_clip = VideoClip(make_frame, duration=total_duration)
        audio = AudioFileClip(audio_path)
        video_clip = video_clip.with_audio(audio)

        seg = os.path.join(TEMP_DIR, f"line_{line['id']}.mp4")
        print(f"  [assembler] Rendering line {line['id']} → {seg}")
        video_clip.write_videofile(seg, fps=FPS, codec="libx264", audio_codec="aac", logger=None)

        # Free the frames array for this line before the next one.
        del frames
        segment_paths.append(seg)
        line_duration += total_duration

    if not segment_paths:
        raise RuntimeError("No renderable lines in script")

    out = output_path(script["topic"])
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"  [assembler] Concatenating {len(segment_paths)} segments → {out}")
    _concat_segments(segment_paths, out)

    for p in segment_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    return out