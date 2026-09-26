"""Batch video production: render ``count`` quote videos in one run.

run_batch() is the mass-production entry. Each video is a fresh set of quotes,
so the batch is a loop over create_video() with round-robin voices and
per-video timing plus a failure tally — a dead Groq key or a bad image fetch
should not abandon the whole batch.
"""
import time

from src.services import voice_manager
from src.pipeline.quote_video import create_video


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
            # Keep going: one bad quote batch or network blip should not cost
            # the remaining videos.
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


def list_voices() -> None:
    """Print the enabled voices, for picking --voice."""
    voice_manager.list_voices()
