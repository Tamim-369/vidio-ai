"""Single-topic video production.

The two entry points cover the whole per-video flow:
research (or provided story) -> script -> images -> voiceover -> assembly.
The full step-by-step ordering and timing accounting live here and nowhere
else; batch.py just calls these per topic.
"""
import time

from src.utils.file_helpers import ensure_workspace, cleanup_temp, dump_artifact
from src.agents.research.sources import research
from src.agents.asset.agent import fetch_assets
from src.agents.voice import agent as voice_manager
from src.services.tts import generate_audio
from src.services.video_assembler import assemble
from src.services.youtube_upload import publish_video
from src.pipeline.lab import build_lab_script, speaking_style_for_style
from src.agents.common.llm import LOCAL_MODEL


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
                 script_only: bool = False):
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

    print("\n🖼️  Fetching images...")
    script["lines"] = _timed("assets", fetch_assets, script["lines"], topic=topic, timing=timing)

    print("\n🎙️  Generating voiceover...")
    script["lines"] = _timed("audio", generate_audio, script["lines"], voice=voice_cfg, topic=topic, timing=timing)

    print("\n🎬 Assembling video...")
    output = _timed("assemble", assemble, script, topic=topic, timing=timing)

    # Record the topic as done so it is never regenerated (fuzzy + exact dedup).
    from src.agents.topic.helpers import record_made_video
    record_made_video(topic)

    if publish:
        print("  (publishing...)")
        _timed("publish", publish_video, output, script["topic"], script, timing=timing)
    else:
        print("\n⏭️  Skipping YouTube upload (pass --no-upload to keep it local)")

    cleanup_temp(topic)
    timing["total"] = time.monotonic() - t0
    print(f"\n⏱️  Build time: {_format_timing(timing)}")
    print(f"\n✅ Done! Video saved to: {output}\n")
    return output


def create_video_from_topic(topic: dict, publish: bool = True, voice: str = "", script_only: bool = False):
    """Run the pipeline for a single topic dict from the topic generator.

    Uses the Reddit story content + source URLs as raw data for scripting.
    Caption-only posts (image subs like tankporn/WarshipPorn) fall back to
    fresh web research so the script isn't built from a one-line title.
    """
    story = (topic.get("content") or topic.get("summary") or "").strip()

    if len(story) >= 800:
        urls = "\n".join(f"- {u}" for u in topic.get("source_urls", []))
        raw_data = f"{story}\n\nSources:\n{urls}".strip()
    else:
        print("  (thin story — researching from the web instead)")
        raw_data = research(topic["title"], urls=topic.get("source_urls") or [])

    return create_video(topic["title"], raw_data=raw_data, publish=publish, voice=voice,
                        script_only=script_only)