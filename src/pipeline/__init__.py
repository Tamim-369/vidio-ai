"""Video-production orchestration — the flows the CLI (src/main.py) signs up to.

Public API:
    create_video(topic, raw_data, publish, voice, script_only)  full pipeline for one topic
    create_video_from_topic(topic, publish, voice, script_only) same, from a topic-generator dict
    run_batch(generate, limit, target, publish, voice, script_only) batch: queue + one video each

script_only=True short-circuits after topic + script generation: assets, audio,
assembly, and upload are skipped and the artifacts are dumped to debug_output/.
"""
from src.pipeline.single import create_video, create_video_from_topic
from src.pipeline.batch import run_batch

__all__ = ["create_video", "create_video_from_topic", "run_batch"]