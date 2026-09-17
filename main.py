import argparse

from src.utils.file_helpers import ensure_dirs, cleanup_temp
from src.services.data_source import research
from src.services.script_builder import build_script
from src.services.asset_fetcher import fetch_assets
from src.services.tts import generate_audio
from src.services.video_assembler import assemble
from src.services.youtube_upload import publish_video
from src.services import voice_manager


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

    voice_id, voice_cfg = voice_manager.pick_voice(preferred=voice)
    style = voice_manager.get_writing_style(voice_id, voice_cfg)
    print(f"\n🗣️  Voice: {voice_cfg['name']} ({voice_id}) — style: {style['name']}")

    ensure_dirs()

    if raw_data:
        print(f"\n📚 Using provided story for: {topic}")
    else:
        print(f"\n🔍 Researching: {topic}")
        raw_data = research(topic)

    print(f"\n📝 Building script...")
    script = build_script(topic, raw_data, style=style)
    print(f"   {len(script['lines'])} lines generated")

    print(f"\n🖼️  Fetching images...")
    script["lines"] = fetch_assets(script["lines"], topic=topic)

    print(f"\n🎙️  Generating voiceover...")
    script["lines"] = generate_audio(script["lines"], voice=voice_cfg)

    print(f"\n🎬 Assembling video...")
    output = assemble(script)

    # Record the topic as done so it is never regenerated (fuzzy + exact dedup).
    from src.services.topic_generator import record_made_video
    record_made_video(topic)

    if publish:
        publish_video(output, script["topic"], script)
    else:
        print("\n⏭️  Skipping YouTube upload (pass --no-upload to keep it local)")

    cleanup_temp()
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
    """Generate (via the research pipeline) or load topics, then produce a video each."""
    from src.services.research_pipeline import run_research_pipeline
    from src.services.topic_generator import load_latest_topics

    if generate:
        print(f"\n🎯 Researching {target} topics...")
        topics = run_research_pipeline(target=target)
    else:
        print("\n📂 Loading latest topic batch...")
        topics = load_latest_topics()

    if not topics:
        print("❌ No topics available. Run without --use-saved to generate new ones.")
        return

    voice_manager.list_voices()

    print(f"\n🎬 Processing {len(topics)} topics...")
    for i, topic in enumerate(topics, 1):
        print(f"\n{'=' * 60}\n[{i}/{len(topics)}] {topic['title']}")
        try:
            # Round-robin across enabled voices: trim → arnold → trim → arnold...
            vid, _ = voice_manager.pick_voice(preferred=voice)
            create_video_from_topic(topic, publish=publish, voice=vid)
        except Exception as e:
            print(f"❌ Failed on topic {i}: {e}")


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