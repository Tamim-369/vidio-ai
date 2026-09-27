"""Make quote videos, one at a time.

    character -> quote (Groq) -> voiceover -> quote card -> music -> upload

A video is one character saying one quote. That is the whole idea: the quote
card is a still frame, so a second line adds narration and a second card and
nothing else, and the formats that stacked several characters per video bought
variety the channel does not need -- one strong line per character, rotating,
is the format the cards were designed for.

So the pipeline is a loop, not a plan. There is nothing to decide up front
except which character goes next, and ``pick_voice`` already round-robins
through the enabled ones. Each video is independent and writes its own file, so
a failure costs one video rather than the batch.
"""
from __future__ import annotations

import time

from src.agents.publish import publish_video
from src.agents.quotes import generate_quotes
from src.agents.soundtrack import add_background_music
from src.agents.visuals import render as render_quote_cards
from src.agents.voice_cast import get_writing_style, pick_voice
from src.agents.voice_cast.voices import VOICES, get_enabled_voices
from src.agents.voiceover import generate_audio
from src.agents.video.artifacts import cleanup_temp, dump_artifact, ensure_dirs
from src.agents.video.title import build_title


def build_script(character: str, quote) -> dict:
    """The one-line script the render steps consume.

    The quote is spoken in full and typeset as a single wrapped block. Splitting
    it by sentence would give a card per sentence, which is not the intended
    "one big quote on screen" look.
    """
    text = (getattr(quote, "text", quote) or "").strip()
    if not text:
        raise RuntimeError("empty quote")
    cfg = VOICES.get(character) or {}
    return {
        "topic": cfg.get("subject", ""),
        "characters": [character],
        "lines": [{"id": 1, "text": text, "character": character, "voice": cfg}],
        "quotes": [{"text": text}],
    }


def create_video(character: str = "", publish: bool = True,
                 script_only: bool = False) -> str:
    """Build one video for one character and return the path.

    character: which voice id to use ("" = the next one in the rotation).
    script_only: write the quote and stop, skipping audio, cards, music and
    upload. Useful for checking tone without paying for a render.
    """
    if publish is None:
        publish = True

    t0 = time.monotonic()
    timing = {}

    character, cfg = pick_voice(preferred=character)
    subject = cfg.get("subject", "")
    style = get_writing_style(character, cfg)
    print(f"\n🗣️  Voice: {cfg['name']} ({character}) — style: {style['name']}")
    print(f"📌 Subject: {subject}")

    ensure_dirs()

    print("\n📝 Generating 1 joke (Groq)...")
    t = time.monotonic()
    # script_only is the "just show me the output" mode, so it is also where a
    # dropped candidate is worth reporting: otherwise a short result looks like
    # the model being stingy rather than the agent filtering it.
    quote = generate_quotes(n=1, subject=subject, character=character,
                            explain=script_only)[0]
    timing["quotes"] = time.monotonic() - t

    script = build_script(character, quote)
    script["title"] = build_title(quote.text, character)
    print(f"   • {quote.text}")
    dump_artifact("quotes", script, script["topic"])

    if script_only:
        timing["total"] = time.monotonic() - t0
        print(f"\n⏱️  Script-only time: {_fmt(timing)}")
        print("\n✅ Quote only — audio, cards, music and upload were skipped.\n")
        return script

    print("\n🎙️  Generating voiceover...")
    t = time.monotonic()
    script["lines"] = generate_audio(script["lines"])
    timing["audio"] = time.monotonic() - t

    print("\n🃏  Rendering quote card...")
    t = time.monotonic()
    output = render_quote_cards(script)
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
            import os
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


def run_batch(count: int = 5, publish: bool = True, voice: str = "",
              script_only: bool = False) -> list:
    """Make ``count`` videos and return the ones that finished.

    Characters rotate, so a batch spreads across every enabled voice instead of
    hammering whichever one happens to be first. A video that fails is reported
    and skipped: one bad render should cost one video, not the batch.
    """
    enabled = [vid for vid, _ in get_enabled_voices()]
    if voice and voice not in enabled:
        raise ValueError(
            f"unknown or disabled voice {voice!r}; pick one of {enabled}")

    t0 = time.monotonic()
    print(f"\n🎬 Batch: {count} video(s){' — script only' if script_only else ''}")

    made, failed = [], []
    for i in range(1, count + 1):
        print(f"\n{'=' * 62}\n  Video {i}/{count}\n{'=' * 62}")
        video_t0 = time.monotonic()
        try:
            made.append(create_video(character=voice, publish=publish,
                                     script_only=script_only))
        except Exception as e:
            print(f"\n❌ Video {i}/{count} failed: {type(e).__name__}: {e}")
            failed.append(i)
            continue
        print(f"⏱️  Video {i} took {time.monotonic() - video_t0:.0f}s")

    total = time.monotonic() - t0
    print(f"\n{'=' * 62}")
    print(f"✅ Batch finished: {len(made)}/{count} made in {total/60:.1f}m")
    if failed:
        print(f"⚠️  Failed videos: {failed}")
    print(f"{'=' * 62}\n")
    return made


def _fmt(timing: dict) -> str:
    return "  ".join(
        f"{k}={f'{v/60:.1f}m' if v >= 120 else f'{v:.0f}s'}"
        for k, v in timing.items() if v > 0
    )
