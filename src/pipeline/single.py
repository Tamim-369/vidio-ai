"""Single-topic video production.

The two entry points cover the whole per-video flow:
research (or provided story) -> script -> images -> voiceover -> assembly.
The full step-by-step ordering and timing accounting live here and nowhere
else; batch.py just calls these per topic.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

from src.utils.file_helpers import ensure_workspace, cleanup_temp, dump_artifact
from src.agents.research.sources import research
from src.agents.asset.agent import fetch_assets
from src.agents.voice import agent as voice_manager
from src.services.tts import generate_audio
from src.services.video_assembler import assemble
from src.services.youtube_upload import publish_video, generate_metadata
from src.pipeline.lab import build_lab_script, speaking_style_for_style
from src.agents.common.llm import LOCAL_MODEL

_metadata_pool = None
_metadata_pool_lock = threading.Lock()

# Heavy-phase gate: at most this many videos run TTS + assembly at once.
# The cheap phases (research/script/assets) are NOT gated, so while one video
# renders, the next video's low-resource work (LLM + network) runs alongside —
# heavy CPU/rendering never stacks. Tune via env for the host's core count.
HEAVY_SLOTS = max(1, int(os.getenv("HEAVY_SLOTS", "2")))
_heavy_slots = threading.BoundedSemaphore(HEAVY_SLOTS)


def _get_metadata_pool() -> ThreadPoolExecutor:
    """Shared metadata (title/description) pool, lazily created like the TTS
    and assembler pools so a whole batch reuses one executor instead of
    spawning one short-lived pool per video."""
    global _metadata_pool
    if _metadata_pool is None:
        with _metadata_pool_lock:
            if _metadata_pool is None:
                _metadata_pool = ThreadPoolExecutor(max_workers=4)
    return _metadata_pool


def _format_timing(timing: dict) -> str:
    """Render a timing dict as a compact summary of nonzero stages."""
    return "  ".join(
        f"{k}={f'{v/60:.1f}m' if v >= 120 else f'{v:.0f}s'}"
        for k, v in timing.items() if v > 0
    )


def _timed(label: str, step, *args, timing: dict, **kwargs):
    """Run `step(*args, **kwargs)`, timing it, and stamp `timing[label]`."""
    t = time.monotonic()
    result = step(*args, **kwargs)
    timing[label] = time.monotonic() - t
    return result


def create_video(topic: str, raw_data: str = None, publish: bool = True, voice: str = "",
                 script_only: bool = False, gate_heavy: bool = False):
    """Run the full pipeline for a single topic and produce a video.

    If raw_data is provided (e.g. a Reddit story from the topic generator),
    it is used directly for scripting instead of doing fresh research.
    publish: True uploads to YouTube after rendering (default). Pass False
    (--no-upload) to render without publishing.
    voice: optional voice id to force ("" = round-robin picks the next voice).
    script_only: stop right after the script is built and dump it —
    skip assets, audio, assembly, and upload entirely.
    """
    if publish is None:
        publish = True

    t0 = time.monotonic()
    timing = {}

    voice_id, voice_cfg = voice_manager.pick_voice(preferred=voice)
    style = voice_manager.get_writing_style(voice_id, voice_cfg)
    print(f"\n🗣️  Voice: {voice_cfg['name']} ({voice_id}) — style: {style['name']}")

    ensure_workspace(topic)

    if raw_data:
        print(f"\n📚 Using provided story for: {topic}")
    else:
        print(f"\n🔍 Researching: {topic}")
        raw_data = research(topic)
    timing["research"] = time.monotonic() - t0
    dump_artifact("research", raw_data, topic)

    print(f"\n📝 Building script (100% local, Ollama {LOCAL_MODEL})...")
    try:
        script = _timed("script", build_lab_script,
                        topic, str(raw_data or ""),
                        style=speaking_style_for_style(style), timing=timing)
    except Exception as e:
        print(f"   ❌ Local script lab failed: {e}")
        raise
    print(f"   {len(script['lines'])} lines generated")

    if script_only:
        dump_artifact("script", script, topic)
        timing["total"] = time.monotonic() - t0
        print(f"\n⏱️  Script-only time: {_format_timing(timing)}")
        print("\n✅ Script only — assets, audio, video, and upload were skipped.\n")
        return script

    # Kick off title/description generation NOW on a background thread. It only
    # needs the script (never the video), so it runs in parallel with asset
    # fetch / audio / assembly and the upload never waits on the LLM metadata.
    metadata_future = None
    if publish:
        print("   (spawning title & description generation in parallel...)")
        metadata_future = _get_metadata_pool().submit(generate_metadata, topic, script)

    print("\n🖼️  Fetching images...")
    script["lines"] = _timed("assets", fetch_assets, script["lines"], topic=topic, timing=timing)

    print("\n🎙️  Generating voiceover...")
    if gate_heavy:
        print(f"   (waiting for a heavy slot: ≤{HEAVY_SLOTS} video(s) render at once)")
    with _heavy_slots if gate_heavy else nullcontext():
        script["lines"] = _timed("audio", generate_audio, script["lines"], voice=voice_cfg, topic=topic, timing=timing)

        print("\n🎬 Assembling video...")
        output = _timed("assemble", assemble, script, topic=topic, timing=timing)

    # Record the topic as done so it is never regenerated (fuzzy + exact dedup).
    from src.agents.topic.helpers import record_made_video
    record_made_video(topic)

    if publish:
        print("  (publishing...)")
        metadata = metadata_future.result() if metadata_future else None
        _timed("publish", publish_video, output, script["topic"], script,
               metadata=metadata, timing=timing)
    else:
        print("\n⏭️  Skipping YouTube upload (pass --no-upload to keep it local)")

    cleanup_temp(topic)
    timing["total"] = time.monotonic() - t0
    print(f"\n⏱️  Build time: {_format_timing(timing)}")
    print(f"\n✅ Done! Video saved to: {output}\n")
    return output


def create_video_from_topic(topic: dict, publish: bool = True, voice: str = "", script_only: bool = False,
                            gate_heavy: bool = False):
    """Run the pipeline for a single topic dict from the topic generator.

    Uses the Reddit story content + source URLs as raw data for scripting.
    Caption-only posts (image subs like tankporn/WarshipPorn) fall back to
    fresh web research so the script isn't built from a one-line title.
    gate_heavy: apply the render-phase gate (HEAVY_SLOTS). Batch passes
    True only when it is producing more than one video; a single video runs
    ungated.
    """
    story = (topic.get("content") or topic.get("summary") or "").strip()

    if len(story) >= 800:
        urls = "\n".join(f"- {u}" for u in topic.get("source_urls", []))
        raw_data = f"{story}\n\nSources:\n{urls}".strip()
    else:
        print("  (thin story — researching from the web instead)")
        raw_data = research(topic["title"], urls=topic.get("source_urls") or [])

    return create_video(topic["title"], raw_data=raw_data, publish=publish, voice=voice,
                        script_only=script_only, gate_heavy=gate_heavy)