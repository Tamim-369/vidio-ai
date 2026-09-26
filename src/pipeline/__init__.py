"""Video-production orchestration — the flows the CLI (src/main.py) signs up to.

Public API:
    create_video(n_quotes, publish, voice, script_only)  full pipeline for one quote video
    run_batch(count, publish, voice, script_only)         batch: N quote videos in one run

script_only=True short-circuits right after the quotes are generated: audio,
quote cards, music and upload are skipped and the artifact is dumped to
debug_output/.
"""
from src.pipeline.quote_video import build_script, create_video
from src.pipeline.batch import run_batch

__all__ = ["build_script", "create_video", "run_batch"]
