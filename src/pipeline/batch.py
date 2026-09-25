"""Batch video production: pick a topic queue and render one video per topic.

run_batch() is the mass-production entry. It splits cleanly into two
self-contained phases:

- _select_topics(): decide WHERE the topic queue comes from — fresh (topic
  agent -> light idea pass -> deep research, each only when the previous
  produced nothing) or the last saved batch.
- _produce_topics(): render every selected topic through
  create_video_from_topic(), round-robining voices, with per-topic timing.
"""
import time

from src.utils.file_helpers import dump_artifact
from src.services import voice_manager
from src.pipeline.single import create_video_from_topic


def _select_topics(generate: bool, target: int) -> list:
    """Resolve the topic queue for this batch run (fresh or saved batch).

    generate=True runs the fallback chain: topic agent first, then the light
    idea pass, then the deep research pipeline — each tried only if the
    previous one came back empty. generate=False loads the latest saved batch.
    """
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
    return topics


def _produce_topics(topics: list, publish: bool, voice: str, script_only: bool = False) -> None:
    """Render one video per topic (or build only scripts), reporting timing."""
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
            create_video_from_topic(topic, publish=publish, voice=vid, script_only=script_only)
        except Exception as e:
            print(f"❌ Failed on topic {i}: {e}")
        per_video.append((topic["title"], time.monotonic() - v_t0))

    batch_el = time.monotonic() - batch_t0
    print(f"\n{'=' * 60}\n⏱️  Batch: {batch_el/60:.1f}m total across {len(per_video)} topic(s)")
    for title, dt in per_video:
        print(f"    {dt/60:5.1f}m  {title[:64]}")
    print("=" * 60)


def run_batch(generate: bool = True, limit: int = 100, target: int = 24, publish: bool = True,
              voice: str = "", script_only: bool = False):
    """Pick the first `target` fresh topics (no ranking) and produce a video each."""
    topics = _select_topics(generate=generate, target=target)

    dump_artifact("topics", topics)

    if not topics:
        print("❌ No topics available. Run without --use-saved to generate new ones.")
        return

    _produce_topics(topics, publish=publish, voice=voice, script_only=script_only)