"""Make quote videos, one at a time.

    character -> quote (Groq) -> voiceover -> quote card -> music -> upload

A video is a *slot in a plan*, not a single character saying a single line. A
batch plans its slots up front (see :mod:`src.agents.video.plan`) so each video
knows its shape before any token is spent: one character with three quotes, or
two or three characters with one quote each. Everything downstream already
handled a list of lines -- the TTS layer groups lines by character so each
voice's conditionals load once, and the card renderer emits one card per line
and concatenates them -- so the plan is the only new thing here.

The plan exists because deciding shape per video as it went produced runs that
looked identical. Withdrawing an earlier rule: a batch used to be one character
saying one quote, on the argument that a second line only adds narration and a
second card. That is true of the *card*, which is a still frame, and false of
the video, which is a sequence of them. Variety across a batch comes from
changing shape, not from only ever shipping one shape.

The pipeline is still a loop, not a plan: a failure costs one video rather than
the batch, and each video writes its own file.
"""
from __future__ import annotations

import time

from src.agents.publish import UploadRejected, publish_video
from src.agents.quotes import generate_quotes
from src.agents.soundtrack import add_background_music
from src.agents.visuals import render as render_quote_cards
from src.agents.voice_cast import get_writing_style
from src.agents.voice_cast.voices import VOICES, get_enabled_voices
from src.agents.voiceover import generate_audio
from src.agents.video.artifacts import cleanup_temp, dump_artifact, ensure_dirs
from src.agents.video.numbering import advance as advance_number, peek as peek_number
from src.agents.video.plan import VideoPlan, plan_batch, with_lead
from src.agents.video.title import build_numbered_title


def build_script(plan: VideoPlan, quotes_by_character: dict, number: int = 0) -> dict:
    """The script the render steps consume, one line per quote.

    Lines are numbered in speak order and carry their own voice config, which
    is what lets the TTS layer group them by character rather than assuming one
    narrator for the whole video. ``topic`` is the lead character's subject,
    since the lead is the one the video is named for.
    """
    lines, quotes = [], []
    for character in plan.characters:
        cfg = VOICES.get(character) or {}
        for quote in quotes_by_character.get(character) or []:
            text = (getattr(quote, "text", quote) or "").strip()
            if not text:
                continue
            lines.append({
                "id": len(lines) + 1,
                "text": text,
                "character": character,
                "voice": cfg,
            })
            quotes.append({"text": text})
    if not lines:
        raise RuntimeError("no usable quotes for this video")

    lead_cfg = VOICES.get(plan.lead) or {}
    script = {
        "topic": lead_cfg.get("subject", ""),
        "characters": list(plan.characters),
        "type_id": plan.type_id,
        "number": number,
        "lines": lines,
        "quotes": quotes,
    }
    if number:
        script["title"] = build_numbered_title(lines[0]["text"], number)
    return script


def create_video(plan: VideoPlan, number: int = 0, publish: bool = True,
                 script_only: bool = False) -> str:
    """Build one video for one planned slot and return the path.

    number: this video's place in the channel's numbering. 0 means unnumbered,
    which is what a single ad-hoc build wants.

    script_only: write the script and stop, skipping audio, cards, music and
    upload. Useful for checking tone without paying for a render.
    """
    if publish is None:
        publish = True

    t0 = time.monotonic()
    timing = {}

    print(f"\n🎬 {plan}")
    lead_cfg = VOICES.get(plan.lead) or {}
    style = get_writing_style(plan.lead, lead_cfg)
    print(f"🗣️  Lead voice: {lead_cfg.get('name')} ({plan.lead}) — style: {style['name']}")
    print(f"📌 Subject: {lead_cfg.get('subject', '')}")
    if number:
        print(f"🔢 Number: #{number}")

    ensure_dirs()

    # One call per character, not per quote. generate_quotes already asks for
    # several candidates in a single request and keeps the first n that pass, so
    # three quotes for one character is still one call. The calls cannot be
    # merged across characters: each voice declares its own subject and has its
    # own prompt module, and the joke has to match the voice saying it.
    quotes_by_character = {}
    t = time.monotonic()
    for character in plan.characters:
        cfg = VOICES.get(character) or {}
        n = plan.quotes_each
        # script_only is the "just show me the output" mode, so it is also where
        # a dropped candidate is worth reporting: otherwise a short result looks
        # like the model being stingy rather than the agent filtering it.
        quotes_by_character[character] = generate_quotes(
            n=n, subject=cfg.get("subject", ""), character=character,
            explain=script_only)
    timing["quotes"] = time.monotonic() - t

    script = build_script(plan, quotes_by_character, number)
    for line in script["lines"]:
        who = (line.get("voice") or {}).get("pseudonym", line["character"])
        print(f"   • [{who}] {line['text']}")
    if not script.get("title"):
        script["title"] = build_numbered_title(script["lines"][0]["text"], number or 1)
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

    print("\n🃏  Rendering quote cards...")
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


def _attempt(plan: VideoPlan, number: int, publish: bool, script_only: bool,
             attempts: int, label: str):
    """Build one planned slot, retrying on a fresh draw. Returns the video, or None.

    A video is retried up to ``attempts`` times because the expensive step is
    neural TTS and a single call can fail on its own -- an unlucky quote, a
    transient model error, memory pressure. The retry re-runs the whole build, so
    it also draws fresh quotes, and quote generation deduplicates against
    everything already used, so a retry cannot repeat the ones that failed.

    The number is spent only on success. A build that never finished put no video
    on the channel, so it must leave the number free for the retry to reuse --
    which is also what keeps a failed slot's type and cast reserved.
    """
    for attempt in range(1, attempts + 1):
        try:
            video = create_video(plan=plan, number=number, publish=publish,
                                 script_only=script_only)
            advance_number()
            return video
        except Exception as e:
            # A failed build can leave a half-written wav or card behind, and the
            # next attempt would otherwise render on top of it.
            try:
                cleanup_temp()
            except Exception:
                pass
            detail = f"{type(e).__name__}: {e}"
            if isinstance(e, UploadRejected):
                # The API rejected the request itself. A fresh quote would be
                # rejected the same way, so stop instead of burning attempts.
                print(f"\n❌ {label} rejected by YouTube "
                      f"(HTTP {e.status}, reason={e.reason or 'unknown'})")
                print("   Not retrying -- fix the request, then re-run.")
                return None
            if attempt < attempts:
                print(f"\n⚠️  Attempt {attempt}/{attempts} failed ({detail})"
                      f"\n   Retrying with a fresh quote...")
            else:
                print(f"\n❌ {label} failed after {attempts} attempt(s): {detail}")
                return None
    return None


def build_one(publish: bool = True, voice: str = "", script_only: bool = False,
              attempts: int = 2):
    """Make a single video.

    Takes the next free slot in the plan, so one video is not a special case
    that happens to be a monologue -- it is whichever shape the channel is due
    next, which is what keeps a lone run and a batch looking like the same feed.
    """
    enabled = [vid for vid, _ in get_enabled_voices()]
    if voice and voice not in enabled:
        raise ValueError(
            f"unknown or disabled voice {voice!r}; pick one of {enabled}")

    number = peek_number()
    plan = plan_batch(1, enabled, start=number - 1)[0]
    if voice:
        plan = with_lead(plan, voice)
    print(f"\n🎬 {plan}")
    return _attempt(plan, number, publish, script_only, max(1, attempts),
                    f"Video #{number}" if number else "Video")


def run_batch(count: int = 5, publish: bool = True, voice: str = "",
              script_only: bool = False, attempts: int = 2) -> list:
    """Make ``count`` videos and return the ones that finished.

    The whole batch is planned before the first video is built, so the shapes
    and casts are known up front and the schedule is inspectable without
    rendering anything.

    The channel number is also the plan's phase. Starting a batch at
    ``number - 1`` means two consecutive runs do not replay the same opening
    types and casts, and a video that failed keeps its slot: because a failed
    build does not advance the counter, the next batch re-enters that video's
    type and character rather than skipping past it.

    A video that fails every attempt is reported and skipped: one bad render
    should cost one video, not the batch.
    """
    enabled = [vid for vid, _ in get_enabled_voices()]
    if voice and voice not in enabled:
        raise ValueError(
            f"unknown or disabled voice {voice!r}; pick one of {enabled}")

    attempts = max(1, attempts)
    t0 = time.monotonic()
    print(f"\n🎬 Batch: {count} video(s){' — script only' if script_only else ''}")

    first_number = peek_number()
    plans = plan_batch(count, enabled, start=first_number - 1)
    if voice:
        plans = [with_lead(p, voice) for p in plans]

    made, failed = [], []
    for i, plan in enumerate(plans, start=1):
        number = peek_number()
        label = f"Video {i}/{count} (number #{number})" if number else f"Video {i}/{count}"
        print(f"\n{'=' * 62}\n  {label}\n{'=' * 62}")
        video_t0 = time.monotonic()
        video = _attempt(plan, number, publish, script_only, attempts, label)
        if video is None:
            failed.append(i)
        else:
            made.append(video)
            print(f"⏱️  Video {i} took {time.monotonic() - video_t0:.0f}s")

    total = time.monotonic() - t0
    print(f"\n{'=' * 62}")
    print(f"✅ Batch finished: {len(made)}/{count} made in {total/60:.1f}m")
    if failed:
        print(f"⚠️  Failed videos: {failed}")
    print(f"📚 Next video number: #{peek_number()}")
    print(f"{'=' * 62}\n")
    return made


def _fmt(timing: dict) -> str:
    return "  ".join(
        f"{k}={f'{v/60:.1f}m' if v >= 120 else f'{v:.0f}s'}"
        for k, v in timing.items() if v > 0
    )
