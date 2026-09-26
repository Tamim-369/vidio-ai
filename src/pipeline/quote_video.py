"""Single-quote video production.

The per-video flow, in order, with all timing accounted for here:

    voice -> quotes (Groq) -> voiceover -> quote cards -> music -> upload

Each quote becomes ONE narration line and ONE still card: the speaker's photo is
the full-bleed background (face bottom-right), the quote is typeset large in the
top 40%, and the fake author is credited bottom-left. There is no image search
and no stock footage — the card is the visual.

Card length follows the spoken audio exactly (TTS duration), so the video is as
long as it takes to say the quote(s) and nothing else.

This replaces the old topic -> research -> script documentary flow and the
intermediate stock-image fetch, which the quote card makes unnecessary.
"""
from __future__ import annotations

import os
import time

from src.services import voice_manager
from src.services.music import add_background_music
from src.services.quote_agent import generate_quotes
from src.services.quote_card import render as render_quote_cards
from src.services.tts import generate_audio
from src.services.youtube_upload import publish_video
from src.utils.file_helpers import cleanup_temp, dump_artifact, ensure_dirs


def build_script(quotes: list) -> dict:
    """Turn quotes into the script dict the render steps consume.

    One quote = one line = one card, so a quote's full text is spoken as a
    single narration line and typeset as a single wrapped block. Splitting by
    sentence would produce a separate card per sentence, which is not the
    intended "one big quote on screen" look.
    """
    lines = []
    for quote in quotes:
        text = (getattr(quote, "text", str(quote)) or "").strip()
        if not text:
            continue
        lines.append({"id": len(lines) + 1, "text": text})
    if not lines:
        raise RuntimeError("quotes produced no narration lines")

    return {
        # The first quote doubles as the video's topic: it is what the video is
        # about, and what gets titled and uploaded.
        "topic": lines[0]["text"],
        "quotes": [
            {"text": getattr(q, "text", str(q)),
             "format": getattr(q, "format", ""),
             "source": getattr(q, "source", ""),
             "figure": getattr(q, "figure", "")}
            for q in quotes
        ],
        "lines": lines,
    }


def create_video(n_quotes: int = 2, publish: bool = True, voice: str = "",
                 script_only: bool = False) -> str:
    """Build one quote video and return the final path.

    voice: force a voice id ("" = round-robin picks the next one).
    script_only: stop after the quotes are written, skipping audio, cards,
    music and upload.
    """
    if publish is None:
        publish = True

    t0 = time.monotonic()
    timing = {}

    voice_id, voice_cfg = voice_manager.pick_voice(preferred=voice)
    style = voice_manager.get_writing_style(voice_id, voice_cfg)
    print(f"\n🗣️  Voice: {voice_cfg['name']} ({voice_id}) — style: {style['name']}")

    ensure_dirs()

    print(f"\n📝 Generating quotes (Groq, {n_quotes} beat)...")
    t = time.monotonic()
    quotes = generate_quotes(n=n_quotes)
    script = build_script(quotes)
    for quote in quotes:
        print(f"   [{quote.format}] {quote.text}")
    print(f"   {len(script['lines'])} card(s) from {len(quotes)} quote(s)")
    timing["quotes"] = time.monotonic() - t
    dump_artifact("quotes", script, script["topic"])

    if script_only:
        timing["total"] = time.monotonic() - t0
        print(f"\n⏱️  Script-only time: {_fmt(timing)}")
        print("\n✅ Quotes only — audio, cards, music and upload were skipped.\n")
        return script

    print("\n🎙️  Generating voiceover...")
    t = time.monotonic()
    script["lines"] = generate_audio(script["lines"], voice=voice_cfg)
    timing["audio"] = time.monotonic() - t

    print("\n🃏  Rendering quote cards...")
    t = time.monotonic()
    output = render_quote_cards(script, voice_cfg)
    timing["cards"] = time.monotonic() - t

    print("\n🎵 Mixing background music...")
    t = time.monotonic()
    cards_only = output
    output = add_background_music(output)
    timing["music"] = time.monotonic() - t

    # The card render is an intermediate, not a deliverable. Leaving it next to
    # the final file means two videos per run, and the music-less one is easy to
    # open by mistake and looks like the music mix silently failed.
    if output != cards_only:
        try:
            os.remove(cards_only)
        except OSError:
            pass

    if publish:
        t = time.monotonic()
        publish_video(output, script["topic"], script)
        timing["publish"] = time.monotonic() - t
    else:
        print("\n⏭️  Skipping YouTube upload (pass --no-upload to keep it local)")

    cleanup_temp()
    timing["total"] = time.monotonic() - t0
    print(f"\n⏱️  Build time: {_fmt(timing)}")
    print(f"\n✅ Done! Video saved to: {output}\n")
    return output


def _fmt(timing: dict) -> str:
    return "  ".join(
        f"{k}={f'{v/60:.1f}m' if v >= 120 else f'{v:.0f}s'}"
        for k, v in timing.items() if v > 0
    )
