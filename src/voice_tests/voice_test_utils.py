#!/usr/bin/env python3
"""Shared test harness for per-voice test videos.

Instead of re-running the whole main pipeline (research + LLM script gen +
asset fetch + upload), each voice test file:

1. GENERATES its test script ONCE with the local script lab
   (src/services/script_lab.build_lab_script + the voice's theme) and
   saves it (plus images, when fetched) into this folder.
2. On every later run, REUSES the cached script so only TTS rendering
   (chatterbox) and video assembly run — no research, no LLM, no downloads.

Voice tests are about the VOICE and captions, not the photography, so:
- lines without usable images get a labeled placeholder frame (assembly
  never crashes, no vision-API budget spent);
- real image fetching is opt-in with `--fetch-images` (it hits the slow,
  rate-limited Groq vision verifier and can reject nearly every query).

Layout under src/voice_tests/:
    scripts/<voice_id>.json   saved structured script (search terms, etc.)
    assets/<voice_id>/        persisted images (or placeholders) per line
    output/<voice_id>.mp4     final rendered test video
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# Make project root importable regardless of cwd.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config.voices import get_voice
from src.config.writing_styles import get_style
from src.services.data_source import research
from src.services.script_lab import build_lab_script, theme_for_style
from src.services.asset_fetcher import fetch_assets
from src.services.tts import generate_audio, _clean_text
from src.services.video_assembler import assemble

VOICE_TESTS = Path(__file__).resolve().parent
SCRIPTS_DIR = VOICE_TESTS / "scripts"
ASSETS_DIR = VOICE_TESTS / "assets"
AUDIO_DIR = VOICE_TESTS / "audio"
OUTPUT_DIR = VOICE_TESTS / "output"


def _voice_paths(voice_id: str):
    script = SCRIPTS_DIR / f"{voice_id}.json"
    assets = ASSETS_DIR / voice_id
    audio = AUDIO_DIR / voice_id
    output = OUTPUT_DIR / f"{voice_id}.mp4"
    return script, assets, audio, output


def _save_script(script: dict, voice_id: str) -> None:
    import json
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_voice_paths(voice_id)[0], "w") as f:
        json.dump(script, f, indent=2)


def _load_script(voice_id: str) -> dict:
    import json
    with open(_voice_paths(voice_id)[0]) as f:
        return json.load(f)


def _persist_assets(script: dict, voice_id: str) -> None:
    """Copy each line's downloaded images into a stable per-voice cache and
    rewrite asset_paths to point at the copies (temp/ is wiped between runs)."""
    _, assets_dir, _, _ = _voice_paths(voice_id)
    assets_dir.mkdir(parents=True, exist_ok=True)

    for line in script["lines"]:
        src_paths = line.get("asset_paths") or ([line["asset_path"]] if line.get("asset_path") else [])
        kept = []
        for i, src in enumerate(src_paths):
            if not src or not os.path.exists(src):
                continue
            ext = os.path.splitext(src)[1] or ".jpg"
            dest = assets_dir / f"line_{line['id']}_{i}{ext}"
            shutil.copy2(src, dest)
            kept.append(str(dest))
        line["asset_paths"] = kept
        line["asset_path"] = kept[0] if kept else None


def _placeholder_assets(line: dict, assets_dir: Path) -> list:
    """Ensure a line has at least one locally-renderable image. Lines that
    fetched nothing get a dark labeled placeholder so assembly can proceed.

    Uses 3 frames per line (same as the real pipeline) so the video looks like
    a normal render: a dark slate + the line's search term + line number.
    """
    existing = [str(p) for p in sorted(
        assets_dir.glob(f"line_{line['id']}_*"),
        key=lambda p: p.suffix == ".png",  # placeholders last only if real imgs exist
    )]
    if existing:
        line["asset_paths"] = existing
        line["asset_path"] = existing[0]
        return existing

    try:
        import textwrap
        from PIL import Image, ImageDraw, ImageFont
        from src.config.settings import VIDEO_RESOLUTIONS, VIDEO_FORMAT, FONT_PATH

        w, h = VIDEO_RESOLUTIONS[VIDEO_FORMAT]
        label = (line.get("search_term") or line.get("text") or f"Line {line['id']}")[:140]

        paths = []
        for slot in range(3):
            img = Image.new("RGB", (w, h), (16, 17, 22))
            d = ImageDraw.Draw(img)
            try:
                font_big = ImageFont.truetype(FONT_PATH, int(h * 0.045))
                font_small = ImageFont.truetype(FONT_PATH, int(h * 0.03))
            except Exception:
                font_big = font_small = ImageFont.load_default()

            d.text((int(w * 0.06), int(h * 0.08)), f"Line {line['id']}",
                   font=font_big, fill=(255, 255, 255))
            lines_wrapped = textwrap.wrap(label, width=38)
            y = int(h * 0.52)
            for ln in lines_wrapped[:6]:
                d.text((int(w * 0.06), y), ln, font=font_small, fill=(200, 210, 230))
                y += int(h * 0.045)

            path = assets_dir / f"line_{line['id']}_{slot}_placeholder.png"
            img.save(path)
            paths.append(str(path))

        line["asset_paths"] = paths
        line["asset_path"] = paths[0]
        return paths
    except Exception as e:
        print(f"  [assets] Placeholder generation failed for line {line['id']}: {e}")
        line["asset_paths"] = []
        line["asset_path"] = None
        return []


def generate_script(voice_id: str, topic: str, fetch_images: bool = False) -> dict:
    """Run research + real LLM script generation for this voice and save it.
    Images are fetched only when fetch_images=True; otherwise placeholders are
    used so the render is fast and burns no vision-API budget."""
    voice = get_voice(voice_id)
    if not voice:
        raise SystemExit(f"Unknown voice '{voice_id}'. Check src/config/voices.py")
    style = get_style(voice.get("writing_style"))

    print(f"\n🗣️  Generating test script for: {voice['name']} ({voice_id})")
    print(f"   Writing style: {style['name']}")

    print(f"\n🔍 Researching: {topic}")
    raw_data = research(topic)

    print("\n📝 Building script (local Ollama script lab)...")
    script = build_lab_script(
        topic,
        str(raw_data or ""),
        theme=theme_for_style(style),
    )
    script.setdefault("topic", topic)
    print(f"   {len(script['lines'])} lines generated")

    _, assets_dir, _, _ = _voice_paths(voice_id)
    assets_dir.mkdir(parents=True, exist_ok=True)

    if fetch_images:
        print("\n🖼️  Fetching images (one-time, slow — vision verify)...")
        script["lines"] = fetch_assets(script["lines"], topic=topic)
        _persist_assets(script, voice_id)

    for line in script["lines"]:
        _placeholder_assets(line, assets_dir)  # no-op for lines that have images

    _save_script(script, voice_id)
    print(f"\n💾 Saved: {_voice_paths(voice_id)[0]}")
    return script


def _audio_cache(script: dict, voice_id: str, force: bool = False) -> None:
    """Generate (or reuse) per-voice TTS voiceover into a persistent cache.

    Mutates `script["lines"]` in place with audio_path/actual_duration. Audio
    is keyed on the CLEANED text (so number/punctuation cleanups invalidate it)
    and lives in src/voice_tests/audio/<voice_id>/ — unlike temp/, which is wiped
    between runs. TTS (chatterbox on CPU) is the slow 95% of a render, so a
    voice-only retest skips it entirely.

    force=True regenerates every line regardless of the cache.
    """
    import soundfile as sf

    POST_VERSION = "12"  # watermark disabled + _declick crackle removal added

    _, _, audio_dir, _ = _voice_paths(voice_id)
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = audio_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def _key(text: str, is_first: bool = False) -> str:
        voice = get_voice(voice_id) or {}
        engine = voice.get("engine", "pocket")
        params = voice.get("params", {})
        ref = voice.get("ref_audio", "")
        # Parameters + reference audio in the key: any voice tweak (exaggeration,
        # temperature, gain...) or a changed ref now invalidates the cache.
        return hashlib.sha256(
            (POST_VERSION + "|" + voice_id + "|" + engine + "|" + ref
             + "|" + json.dumps(params, sort_keys=True)
             + "|" + ("F1" if is_first else "")
             + "|" + _clean_text(text)).encode()
        ).hexdigest()[:16]

    need = []
    for i, line in enumerate(script["lines"]):
        line_id = str(line["id"])
        line["is_first"] = (i == 0)  # first line gets the lead-word buffer in TTS
        wav = audio_dir / f"line_{line_id}.wav"
        key = _key(line["text"], is_first=line["is_first"])
        if not force and manifest.get(line_id) == key and wav.exists():
            info = sf.info(str(wav))
            line["audio_path"] = str(wav)
            line["actual_duration"] = info.frames / info.samplerate
        else:
            need.append(line)

    if need:
        print(f"\n🎙️  Regenerating voiceover (TTS, {len(need)} line(s))...")
        generated = generate_audio(need, voice=get_voice(voice_id))
        for g in generated:
            line_id = str(g["id"])
            wav = audio_dir / f"line_{line_id}.wav"
            shutil.copy2(g["audio_path"], wav)
            g["audio_path"] = str(wav)
            manifest[line_id] = _key(g["text"], is_first=g.get("is_first", False))
        manifest_path.write_text(json.dumps(manifest, indent=2))
    else:
        print("\n🎙️  Voiceover already rendered — reusing cached audio (TTS skipped)")


def render_video(voice_id: str, force_audio: bool = False) -> str:
    """Render a test video for `voice_id` from its cached script + cached audio.
    Re-runs only assembly (no research/LLM/downloads/TTS unless needed)."""
    voice = get_voice(voice_id)
    if not voice:
        raise SystemExit(f"Unknown voice '{voice_id}'. Check src/config/voices.py")

    script, assets_dir, _, output = _voice_paths(voice_id)
    if not script.exists():
        print(f"❌ No cached script at {script}. Run with `--topic` first.")
        raise SystemExit(1)

    print(f"\n🗣️  Rendering test video for: {voice['name']} ({voice_id})")

    # Ensure script + placeholder assets are in place for this render.
    loaded = _load_script(voice_id)
    for line in loaded["lines"]:
        _placeholder_assets(line, assets_dir)

    # TTS only when invalidated/forced; otherwise reused from the cache.
    _audio_cache(loaded, voice_id, force=force_audio)

    print("\n🎬 Assembling video...")
    final_path = assemble(loaded)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_path, output)
    print(f"\n✅ Test video saved to: {output}")
    return str(output)


def main(voice_id: str, default_topic: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Render a {voice_id} test video from cached script (re-generate with --topic).")
    parser.add_argument("--topic", default="",
                        help="Regenerate the test script for this topic (research + LLM)")
    parser.add_argument("--fetch-images", action="store_true",
                        help="Also fetch + verify real images (slow, burns Groq vision tokens)")
    parser.add_argument("--force-audio", action="store_true",
                        help="Regenerate the TTS voiceover (ignore cached audio)")
    parser.add_argument("--no-render", action="store_true",
                        help="Only generate/save the script, do not render the video")
    args = parser.parse_args()

    if args.topic:
        generate_script(voice_id, args.topic, fetch_images=args.fetch_images)
    elif not _voice_paths(voice_id)[0].exists():
        print(f"ℹ️  No cached script yet — generating one for: {default_topic}")
        generate_script(voice_id, default_topic, fetch_images=args.fetch_images)

    if not args.no_render:
        render_video(voice_id, force_audio=args.force_audio)


if __name__ == "__main__":
    main("donald-trump", "A historical military failure with exact numbers, distances and years")