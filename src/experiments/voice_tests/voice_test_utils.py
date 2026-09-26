#!/usr/bin/env python3
"""Shared test harness for per-voice test videos.

Renders one quote video per voice so the CLONE can be judged by ear without
running the whole pipeline end to end. The visual is the same quote card the
real pipeline uses: the voice's own photo as a full-bleed background, the quote
in the top 40%, and a fake author credited bottom-left.

The narration text comes from the quote agent (src/services/quote_agent.py,
Groq-only), which is the same content source the real pipeline uses. That keeps
the harness honest: if a quote sounds wrong in this voice, it will sound wrong
in production. Each quote is ONE narration line and ONE card, because the card
typesets a whole quote as a single wrapped block.

1. GENERATES the quotes ONCE (unless a cache exists) and saves them under this
   folder, so later runs only re-run TTS + the card render — no Groq calls.
2. Audio is cached per voice, so a voice-only retest skips TTS entirely.

Layout under src/experiments/voice_tests/:
    scripts/<voice_id>.json   saved quotes + narration lines
    audio/<voice_id>/         cached TTS wavs + manifest
    output/<voice_id>.mp4     final rendered test video
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

# Make project root importable regardless of cwd.
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.config.voices import get_voice
from src.config.writing_styles import get_style
from src.services.quote_agent import generate_quotes
from src.services.quote_card import render as render_quote_cards
from src.services.tts import generate_audio, _clean_text

VOICE_TESTS = Path(__file__).resolve().parent
SCRIPTS_DIR = VOICE_TESTS / "scripts"
AUDIO_DIR = VOICE_TESTS / "audio"
OUTPUT_DIR = VOICE_TESTS / "output"


def _voice_paths(voice_id: str):
    script = SCRIPTS_DIR / f"{voice_id}.json"
    audio = AUDIO_DIR / voice_id
    output = OUTPUT_DIR / f"{voice_id}.mp4"
    return script, audio, output


def _save_script(script: dict, voice_id: str) -> None:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_voice_paths(voice_id)[0], "w") as f:
        json.dump(script, f, indent=2)


def _load_script(voice_id: str) -> dict:
    with open(_voice_paths(voice_id)[0]) as f:
        return json.load(f)


def _lines_from_quotes(quotes: list) -> list:
    """One quote = one line = one card (the card wraps the text itself)."""
    lines = []
    for quote in quotes:
        text = (getattr(quote, "text", str(quote)) or "").strip()
        if not text:
            continue
        lines.append({"id": len(lines) + 1, "text": text})
    return lines


def generate_script(voice_id: str, n_quotes: int = 2) -> dict:
    """Fetch fresh quotes for this voice and save them as a renderable script."""
    voice = get_voice(voice_id)
    if not voice:
        raise SystemExit(f"Unknown voice '{voice_id}'. Check src/config/voices.py")
    style = get_style(voice.get("writing_style"))

    print(f"\n🗣️  Generating test quotes for: {voice['name']} ({voice_id})")
    print(f"   Writing style: {style['name']}")

    print("\n📝 Fetching quotes (Groq, quote agent)...")
    quotes = generate_quotes(n=n_quotes)
    for q in quotes:
        print(f"   [{q.format}] {q.text}")

    lines = _lines_from_quotes(quotes)
    if not lines:
        raise SystemExit("quote agent returned no narration lines")

    script = {
        "topic": getattr(quotes[0], "text", ""),
        "quotes": [asdict(q) for q in quotes],
        "lines": lines,
    }
    print(f"   {len(lines)} card(s) from {len(quotes)} quote(s)")
    print(f"   Background: {voice.get('face')}")

    _save_script(script, voice_id)
    print(f"\n💾 Saved: {_voice_paths(voice_id)[0]}")
    return script


def _audio_cache(script: dict, voice_id: str, force: bool = False) -> None:
    """Generate (or reuse) per-voice TTS voiceover into a persistent cache.

    Mutates `script["lines"]` in place with audio_path/actual_duration. Audio
    is keyed on the CLEANED text (so number/punctuation cleanups invalidate it)
    and lives in src/experiments/voice_tests/audio/<voice_id>/ — unlike temp/, which is wiped
    between runs. TTS (chatterbox on CPU) is the slow 95% of a render, so a
    voice-only retest skips it entirely.

    force=True regenerates every line regardless of the cache.
    """
    import soundfile as sf

    POST_VERSION = "12"  # watermark disabled + _declick crackle removal added

    _, audio_dir, _ = _voice_paths(voice_id)
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
    Re-runs only the card render (no LLM calls or TTS unless needed)."""
    voice = get_voice(voice_id)
    if not voice:
        raise SystemExit(f"Unknown voice '{voice_id}'. Check src/config/voices.py")

    script, _, output = _voice_paths(voice_id)
    if not script.exists():
        print(f"❌ No cached script at {script}. Run with --quotes first.")
        raise SystemExit(1)

    print(f"\n🗣️  Rendering test video for: {voice['name']} ({voice_id})")

    loaded = _load_script(voice_id)

    # TTS only when invalidated/forced; otherwise reused from the cache.
    _audio_cache(loaded, voice_id, force=force_audio)

    print("\n🃏  Rendering quote cards...")
    final_path = render_quote_cards(loaded, voice)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_path, output)
    print(f"\n✅ Test video saved to: {output}")
    return str(output)


def main(voice_id: str, default_quotes: int = 2) -> None:
    parser = argparse.ArgumentParser(
        description=f"Render a {voice_id} test video from cached quotes (re-generate with --quotes).")
    parser.add_argument("--quotes", type=int, default=0,
                        help="Force fresh quotes (re-generation) instead of the cache")
    parser.add_argument("--force-audio", action="store_true",
                        help="Regenerate the TTS voiceover (ignore cached audio)")
    parser.add_argument("--no-render", action="store_true",
                        help="Only generate/save the script, do not render the video")
    args = parser.parse_args()

    n_quotes = args.quotes or default_quotes
    if args.quotes:
        generate_script(voice_id, n_quotes=n_quotes)
    elif not _voice_paths(voice_id)[0].exists():
        print(f"ℹ️  No cached quotes yet — generating {n_quotes} fresh quote(s)")
        generate_script(voice_id, n_quotes=n_quotes)

    if not args.no_render:
        render_video(voice_id, force_audio=args.force_audio)


if __name__ == "__main__":
    main("donald-trump", 2)
