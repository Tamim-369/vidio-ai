import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.file_helpers import ensure_dirs, cleanup_temp
from src.services.data_source import research
from src.services.asset_fetcher import fetch_assets
from src.services.tts import generate_audio
from src.services.video_assembler import assemble
from src.services.youtube_upload import publish_video
from src.services import voice_manager
from src.services.script_lab import build_lab_script, theme_for_style


def create_video(topic: str, raw_data: str = None, publish: bool = True, voice: str = ""):
    """Run the full pipeline for a single topic and produce a video.

    If raw_data is provided (e.g. a Reddit story from the topic generator),
    it is used directly for scripting instead of doing fresh research.
    publish: True uploads to YouTube after rendering (default). Pass False
    (--no-upload) to render without publishing.
    voice: optional voice id to force ("" = round-robin picks the next voice).
    """
    if publish is None:
        publish = True

    t0 = time.monotonic()
    timing = {}

    voice_id, voice_cfg = voice_manager.pick_voice(preferred=voice)
    style = voice_manager.get_writing_style(voice_id, voice_cfg)
    print(f"\n🗣️  Voice: {voice_cfg['name']} ({voice_id}) — style: {style['name']}")

    ensure_dirs()

    if raw_data:
        print(f"\n📚 Using provided story for: {topic}")
    else:
        print(f"\n🔍 Researching: {topic}")
        raw_data = research(topic)
    timing["research"] = time.monotonic() - t0

    print(f"\n📝 Building script (100% local, Ollama phi4-mini)...")
    t = time.monotonic()
    try:
        script = build_lab_script(
            topic,
            str(raw_data or ""),
            theme=theme_for_style(style),
        )
    except Exception as e:
        print(f"   ❌ Local script lab failed: {e}")
        raise
    print(f"   {len(script['lines'])} lines generated")
    timing["script"] = time.monotonic() - t

    print(f"\n🖼️  Fetching images...")
    t = time.monotonic()
    script["lines"] = fetch_assets(script["lines"], topic=topic)
    timing["assets"] = time.monotonic() - t

    print(f"\n🎙️  Generating voiceover...")
    t = time.monotonic()
    script["lines"] = generate_audio(script["lines"], voice=voice_cfg)
    timing["audio"] = time.monotonic() - t

    print(f"\n🎬 Assembling video...")
    t = time.monotonic()
    output = assemble(script)
    timing["assemble"] = time.monotonic() - t

    # Record the topic as done so it is never regenerated (fuzzy + exact dedup).
    from src.services.topic_generator import record_made_video
    record_made_video(topic)

    if publish:
        t = time.monotonic()
        publish_video(output, script["topic"], script)
        timing["publish"] = time.monotonic() - t
    else:
        print("\n⏭️  Skipping YouTube upload (pass --no-upload to keep it local)")

    cleanup_temp()
    timing["total"] = time.monotonic() - t0
    parts = "  ".join(
        f"{k}={f'{v/60:.1f}m' if v >= 120 else f'{v:.0f}s'}"
        for k, v in timing.items() if v > 0
    )
    print(f"\n⏱️  Build time: {parts}")
    print(f"\n✅ Done! Video saved to: {output}\n")
    return output


def create_video_from_topic(topic: dict, publish: bool = True, voice: str = "") -> str:
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
        raw_data = research(topic["title"])

    return create_video(topic["title"], raw_data=raw_data, publish=publish, voice=voice)


def run_batch(generate: bool = True, limit: int = 100, target: int = 24, publish: bool = True, voice: str = ""):
    """Pick the first `target` fresh topics (no ranking) and produce a video each."""
    from src.services.research_pipeline import generate_first_topics, run_research_pipeline
    from src.services.topic_generator import load_latest_topics

    if generate:
        print(f"\n🎯 Topic agent: picking {target} fresh topic(s)...")
        try:
            from src.services.topic_agent.agent import run_topic_agent

            topics = run_topic_agent(target=target)
        except Exception as e:
            print(f"⚠️  Topic agent failed ({e}) — falling back")
            topics = []
        if not topics:
            print("Topic agent empty — falling back to light idea pass")
            topics = generate_first_topics(target=target)
        if not topics:
            print("Light idea pass empty — falling back to deep research pipeline")
            topics = run_research_pipeline(target=target)
    else:
        print("\n📂 Loading latest topic batch...")
        topics = load_latest_topics()

    if not topics:
        print("❌ No topics available. Run without --use-saved to generate new ones.")
        return

    voice_manager.list_voices()

    print(f"\n🎬 Processing {len(topics)} topics...")
    batch_t0 = time.monotonic()
    per_video = []
    for i, topic in enumerate(topics, 1):
        print(f"\n{'=' * 60}\n[{i}/{len(topics)}] {topic['title']}")
        v_t0 = time.monotonic()
        try:
            # Round-robin across enabled voices: trim → arnold → trim → arnold...
            vid, _ = voice_manager.pick_voice(preferred=voice)
            create_video_from_topic(topic, publish=publish, voice=vid)
        except Exception as e:
            print(f"❌ Failed on topic {i}: {e}")
        per_video.append((topic["title"], time.monotonic() - v_t0))

    batch_el = time.monotonic() - batch_t0
    print(f"\n{'=' * 60}\n⏱️  Batch: {batch_el/60:.1f}m total across {len(per_video)} topic(s)")
    for title, dt in per_video:
        print(f"    {dt/60:5.1f}m  {title[:64]}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faceless YouTube video pipeline")
    parser.add_argument("topic", nargs="?", default=None, help="Single topic to make a video for")
    parser.add_argument("--batch", action="store_true", help="Generate topics and make videos for all")
    parser.add_argument("--use-saved", action="store_true", help="Use saved topic batch instead of generating")
    parser.add_argument("--upload", action="store_true", help="Upload to YouTube after rendering (default)")
    parser.add_argument("--no-upload", action="store_true", help="Render WITHOUT uploading to YouTube")
    parser.add_argument("--voice", default="", help="Force a specific voice id (see --list-voices)")
    parser.add_argument("--list-voices", action="store_true", help="List all registered voices and exit")
    parser.add_argument("--limit", type=int, default=100, help="Posts per source when researching topics")
    parser.add_argument("--target", type=int, default=24, help="How many topics to research")
    args = parser.parse_args()

    publish = not args.no_upload

    if args.list_voices:
        voice_manager.list_voices()
    elif args.batch:
        run_batch(generate=not args.use_saved, limit=args.limit, target=args.target, publish=publish, voice=args.voice)
    elif args.topic:
        create_video(args.topic, publish=publish, voice=args.voice)
    else:
        # Default: generate topics from Reddit and make videos for all of them
        run_batch(generate=True, limit=args.limit, target=args.target, publish=publish, voice=args.voice)