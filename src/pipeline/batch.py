"""Batch video production: pick a topic queue and render one video per topic.

run_batch() is the mass-production entry. It splits cleanly into two
self-contained phases:

- _select_topics(): decide WHERE the topic queue comes from — fresh (topic
  agent -> light idea pass -> deep research, each only when the previous
  produced nothing) or the last saved batch.
- _produce_topics(): render topics through create_video_from_topic() on a
  moving assembly line: up to `concurrency` videos in flight at once, each in
  its own temp workspace, round-robining voices. The cheap (LLM/network)
  phases of all in-flight videos overlap freely; the heavy render phases
  (TTS + assembly) are gated by HEAVY_SLOTS in single.py so expensive CPU work
  never stacks — the next video's research/script/assets run while the
  current one renders. Serializing instructions land in parallel workers here;
  per-video timing is still reported.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from src.utils.file_helpers import dump_artifact
from src.agents.voice import agent as voice_manager
from src.pipeline.single import create_video_from_topic


def _select_topics(generate: bool, target: int) -> list:
    """Resolve the topic queue for this batch run (fresh or saved batch).

    generate=True runs the fallback chain: topic agent first, then the light
    idea pass, then the deep research pipeline — each tried only if the
    previous one came back empty. generate=False loads the latest saved batch.
    """
    from src.agents.research.agent import generate_first_topics, run_research_pipeline
    from src.agents.topic.helpers import load_latest_topics

    if generate:
        print(f"\n🎯 Topic agent: picking {target} fresh topic(s)...")
        try:
            from src.agents.topic.agent import run_topic_agent

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


def _produce_topics(topics: list, publish: bool, voice: str, script_only: bool = False,
                    concurrency: int = 1) -> None:
    """Render topics on a moving assembly line, reporting per-topic timing.

    Voice assignment happens BEFORE any worker starts so the round-robin cursor
    stays deterministic (pick_voice is not thread-safe). Each worker's
    create_video_from_topic runs in its own temp workspace, so none of them
    collide on temp/ paths even when they are mid-flight at the same time.

    The render-phase gate (single.py's HEAVY_SLOTS) only applies when there
    are more than one video to make: with a single video there is nothing to
    overlap, so it renders without waiting on a slot.
    """
    voice_manager.list_voices()

    print(f"\n🎬 Processing {len(topics)} topics ({concurrency} at a time)...")
    gate_heavy = len(topics) > 1
    assigned = [
        (t, voice_manager.pick_voice(preferred=voice)[0], gate_heavy)
        for t in topics
    ]

    batch_t0 = time.monotonic()
    per_video = []

    def _work(topic_vid_gate):
        topic, vid, gate_heavy = topic_vid_gate
        v_t0 = time.monotonic()
        try:
            create_video_from_topic(topic, publish=publish, voice=vid, script_only=script_only,
                                    gate_heavy=gate_heavy)
        except Exception as e:
            print(f"❌ Failed on topic: {e}")
        return (topic["title"], time.monotonic() - v_t0)

    if concurrency <= 1:
        for t in assigned:
            per_video.append(_work(t))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for result in ex.map(_work, assigned):
                per_video.append(result)

    batch_el = time.monotonic() - batch_t0
    print(f"\n{'=' * 60}\n⏱️  Batch: {batch_el/60:.1f}m total across {len(per_video)} topic(s)")
    for title, dt in per_video:
        print(f"    {dt/60:5.1f}m  {title[:64]}")
    print("=" * 60)


def run_batch(generate: bool = True, limit: int = 100, target: int = 24, publish: bool = True,
              voice: str = "", script_only: bool = False, concurrency: int = 1):
    """Pick the first `target` fresh topics (no ranking) and produce a video each."""
    topics = _select_topics(generate=generate, target=target)

    dump_artifact("topics", topics)

    if not topics:
        print("❌ No topics available. Run without --use-saved to generate new ones.")
        return

    _produce_topics(topics, publish=publish, voice=voice, script_only=script_only,
                    concurrency=concurrency)