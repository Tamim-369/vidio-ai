"""Assemble one quote video, end to end:

    voice -> jokes (Groq) -> voiceover -> quote cards -> music -> upload

The voice is picked first and decides the subject, so the joke and the narrator
always agree. Each joke is ONE narration line and ONE still card, sized to the
spoken audio, so the video is exactly as long as it takes to say it."""
from __future__ import annotations

import os
import time

from src.agents.publish import publish_video
from src.agents.quotes import generate_quotes
from src.agents.soundtrack import add_background_music
from src.agents.visuals import render as render_quote_cards
from src.agents.voice_cast import get_writing_style, pick_voice
from src.agents.voiceover import generate_audio
from src.agents.video.artifacts import cleanup_temp, dump_artifact, ensure_dirs


def build_script(quotes: list, subject: str = "") -> dict:
    """Turn quotes into the script dict the render steps consume.

    One quote = one line = one card, so a quote's full text is spoken as a
    single narration line and typeset as a single wrapped block. Splitting by
    sentence would produce a separate card per sentence, which is not the
    intended "one big quote on screen" look.

    The video's topic is the subject, not the first joke: the subject is what
    the video is actually about, and it is what names the file and titles the
    upload. Using a random joke instead produced filenames and YouTube titles
    that read like the joke itself.
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
        "topic": subject or lines[0]["text"],
        "subject": subject,
        "quotes": [{"text": line["text"]} for line in lines],
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

    voice_id, voice_cfg = pick_voice(preferred=voice)
    style = get_writing_style(voice_id, voice_cfg)
    subject = voice_cfg.get("subject", "")
    print(f"\n🗣️  Voice: {voice_cfg['name']} ({voice_id}) — style: {style['name']}")
    print(f"📌 Subject: {subject}")

    ensure_dirs()

    print(f"\n📝 Generating {n_quotes} joke(s) (Groq, {subject})...")
    t = time.monotonic()
    # script_only is the "just show me the output" mode, so it is also where a
    # dropped candidate is worth reporting: otherwise a short result looks like
    # the model being stingy rather than the agent filtering it.
    quotes = generate_quotes(n=n_quotes, subject=subject, explain=script_only,
                             character=voice_id)
    script = build_script(quotes, subject)
    for quote in quotes:
        print(f"   • {quote.text}")
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


def run_batch(count: int = 5, publish: bool = True, voice: str = "",
              script_only: bool = False) -> list:
    """Render ``count`` videos and return the paths of the ones that finished.

    count: how many videos to produce. publish: upload each to YouTube after
    rendering. voice: force one voice id ("" = round-robin across the enabled
    voices). script_only: generate the quotes only.
    """
    t0 = time.monotonic()
    print(f"\n🎬 Batch: {count} quote video(s)"
          f"{' — script only' if script_only else ''}")

    done = []
    failed = []
    for index in range(1, count + 1):
        print(f"\n{'=' * 62}\n  Video {index}/{count}\n{'=' * 62}")
        video_t0 = time.monotonic()
        try:
            if script_only:
                create_video(publish=False, voice=voice, script_only=True)
                done.append(None)
            else:
                done.append(create_video(publish=publish, voice=voice))
        except Exception as e:
            print(f"\n❌ Video {index}/{count} failed: {type(e).__name__}: {e}")
            failed.append(index)
            continue
        print(f"⏱️  Video {index} took {time.monotonic() - video_t0:.0f}s")

    total = time.monotonic() - t0
    print(f"\n{'=' * 62}")
    print(f"✅ Batch finished: {len(done)}/{count} rendered in {total/60:.1f}m")
    if failed:
        print(f"⚠️  Failed videos: {failed}")
    print(f"{'=' * 62}\n")
    return done
