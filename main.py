import argparse

from src.utils.file_helpers import ensure_dirs, cleanup_temp
from src.services.data_source import research
from src.services.script_builder import build_script
from src.services.asset_fetcher import fetch_assets
from src.services.tts import generate_audio
from src.services.video_assembler import assemble
from src.services.youtube_upload import publish_video
from src.config.settings import AUTO_PUBLISH


def create_video(topic: str, raw_data: str = None, publish: bool = None):
    """Run the full pipeline for a single topic and produce a video.

    If raw_data is provided (e.g. a Reddit story from the topic generator),
    it is used directly for scripting instead of doing fresh research.
    publish=None uses the AUTO_PUBLISH setting from .env.
    """
    if publish is None:
        publish = AUTO_PUBLISH

    ensure_dirs()

    if raw_data:
        print(f"\n📚 Using provided story for: {topic}")
    else:
        print(f"\n🔍 Researching: {topic}")
        raw_data = research(topic)

    print(f"\n📝 Building script...")
    script = build_script(topic, raw_data)
    print(f"   {len(script['lines'])} lines generated")

    print(f"\n🖼️  Fetching images...")
    script["lines"] = fetch_assets(script["lines"], topic=topic)

    print(f"\n🎙️  Generating voiceover...")
    script["lines"] = generate_audio(script["lines"])

    print(f"\n🎬 Assembling video...")
    output = assemble(script)

    if publish:
        publish_video(output, script["topic"], script)
    else:
        print("\n⏭️  Skipping YouTube upload (set AUTO_PUBLISH=1 in .env or use --upload)")

    cleanup_temp()
    print(f"\n✅ Done! Video saved to: {output}\n")
    return output


def create_video_from_topic(topic: dict, publish: bool = None) -> str:
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

    return create_video(topic["title"], raw_data=raw_data, publish=publish)


def run_batch(generate: bool = True, limit: int = 100, target: int = 20, publish: bool = None):
    """Generate (or load) topics and produce a video for each one."""
    from src.services.topic_generator import run_topic_generation, load_latest_topics

    if generate:
        print(f"\n🎯 Generating {target} topics...")
        topics = run_topic_generation(limit=limit, target=target)
    else:
        print("\n📂 Loading latest topic batch...")
        topics = load_latest_topics()

    if not topics:
        print("❌ No topics available. Run without --use-saved to generate new ones.")
        return

    print(f"\n🎬 Processing {len(topics)} topics...")
    for i, topic in enumerate(topics, 1):
        print(f"\n{'=' * 60}\n[{i}/{len(topics)}] {topic['title']}")
        try:
            create_video_from_topic(topic, publish=publish)
        except Exception as e:
            print(f"❌ Failed on topic {i}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faceless YouTube video pipeline")
    parser.add_argument("topic", nargs="?", default=None, help="Single topic to make a video for")
    parser.add_argument("--batch", action="store_true", help="Generate topics and make videos for all")
    parser.add_argument("--use-saved", action="store_true", help="Use saved topic batch instead of generating")
    parser.add_argument("--upload", action="store_true", help="Upload to YouTube after making videos")
    parser.add_argument("--limit", type=int, default=100, help="Posts per subreddit when generating topics")
    parser.add_argument("--target", type=int, default=20, help="How many topics to generate")
    args = parser.parse_args()

    publish = True if args.upload else None

    if args.batch:
        run_batch(generate=not args.use_saved, limit=args.limit, target=args.target, publish=publish)
    elif args.topic:
        create_video(args.topic, publish=publish)
    else:
        # Default: generate topics from Reddit and make videos for all of them
        run_batch(generate=True, limit=args.limit, target=args.target, publish=publish)