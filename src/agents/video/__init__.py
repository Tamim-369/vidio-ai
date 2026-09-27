"""One finished quote video: character -> quote -> voice -> card -> upload."""
from typing import TYPE_CHECKING

__all__ = ["build_script", "create_video", "run_batch"]

if TYPE_CHECKING:  # for type checkers and IDEs; never executed at runtime
    from src.agents.video.pipeline import build_script, create_video, run_batch


def __getattr__(name: str):
    if name in __all__:
        from src.agents.video import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
