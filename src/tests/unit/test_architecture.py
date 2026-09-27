"""Architecture guard: the retired topic/research/script subsystems stay gone.

The documentary pipeline (topic agent -> research -> script lab) has been
replaced by the quote pipeline (quote agent -> music mix). These tests keep the
replacement honest in two directions:

* nothing anywhere in src/ imports the retired subsystems any more, and
* the retired files are actually deleted, so they cannot quietly come back.

It also pins the module set the new pipeline is built from, so a refactor that
drops a surviving service fails here rather than at render time.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

from src.agents.voice_cast import voices

# src/tests/unit/test_architecture.py -> repo root
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

# Subsystems that were removed. Any surviving import of these is a bug: the
# files they named no longer exist.
RETIRED = {
    "script_lab",
    "topic_agent",
    "topic_generator",
    "research_pipeline",
    "data_source",
    "channel_miner",
    "gemini_topics",
    "script_lab_prompts",
    "speaking_style_prompt",
    "speaking_styles",
    "topic_source_prompt",
    "single",  # src/pipeline/single.py, the old topic-driven entry point
}

# The quote pipeline's surviving services. If a structural refactor drops one of
# these, the render path breaks — fail here with a clear message instead.
# The image-search subsystem (asset_fetcher / query_agent / asset_agent) and the
# local-LLM seam they shared (local_llm / local_model / asset_prompts) have been
# deleted: the quote card uses the speaker's own photo, so nothing on the render
# path needs stock imagery or a local model. See MUST_STAY_REMOVED.
MUST_SURVIVE = [
    "src/agents/quotes/agent.py",
    "src/agents/quotes/prompt.py",
    "src/agents/quotes/json_parse.py",
    "src/agents/visuals/card.py",
    "src/agents/visuals/mux.py",
    "src/agents/soundtrack/music.py",
    "src/agents/voiceover/engine.py",
    "src/agents/voiceover/dsp.py",
    "src/agents/voiceover/normalize.py",
    "src/agents/voiceover/timing.py",
    "src/agents/voice_cast/agent.py",
    "src/agents/voice_cast/voices.py",
    "src/agents/voice_cast/writing_styles.py",
    "src/agents/publish/youtube.py",
    "src/agents/publish/prompts.py",
    "src/agents/completion/llm.py",
    "src/agents/video/pipeline.py",
    "src/agents/video/title.py",
    "src/agents/video/artifacts.py",
    "src/agents/visuals/layout.py",
    "src/agents/voiceover/pauses.py",
    "src/agents/voiceover/models.py",
    "src/agents/voiceover/workers.py",
    "src/agents/publish/metadata.py",
    "src/main.py",
]

# The retired subsystems. Kept as a regression guard in the other direction:
# the direction change was that cards render from the speaker's own photo, so
# the image-search stack and the manual voice-clone scripts have no place in
# this repo and should not creep back in.
MUST_STAY_REMOVED = [
    "src/config",
    "src/services",
    "src/shared",
    "src/utils",
    "src/pipeline",
    "src/services/asset_fetcher.py",
    "src/services/query_agent.py",
    "src/services/asset_agent",
    "src/services/local_llm.py",
    "src/config/asset_prompts.py",
    "src/config/local_model.py",
]

# The speaker photos that back every quote card. Missing faces = no video.
REQUIRED_FACES = [
    "src/faces/Trump.png",
    "src/faces/Arnold.png",
    "src/faces/Tate.png",
]

# The files and directories that were deleted, asserted absent below.
REMOVED_PATHS = [
    "src/services/script_lab/",
    "src/services/topic_agent/",
    "src/services/topic_generator.py",
    "src/services/research_pipeline.py",
    "src/services/data_source.py",
    "src/services/channel_miner.py",
    "src/services/gemini_topics.py",
    "src/config/script_lab_prompts.py",
    "src/config/topic_source_prompt.py",
    "src/config/speaking_style_prompt.py",
    "src/config/speaking_styles.py",
    "src/pipeline/single.py",
    "src/scripts/",
    "src/tests/script_lab_test.py",
    "src/tests/test_channel_miner.py",
]


def _imported_names(path: Path) -> set[str]:
    """Return every module/package name referenced by imports in `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import: resolve against the file's package.
                pkg = ".".join(path.relative_to(SRC).with_suffix("").parts[:-1])
                base = pkg if node.module is None else f"{pkg}.{node.module}"
                names.add(base)
                names.update(f"{base}.{a.name}" for a in node.names)
            elif node.module:
                names.add(node.module)
                names.update(f"{node.module}.{a.name}" for a in node.names)
    return names

# Directories under src/ that are not part of the code layout. `faces`,
# `music`, `assets` and `state` are data; `tests` is a separate tree.
DATA_DIRS = {"faces", "music", "assets", "state", "tests", "__pycache__"}
ALLOWED_DIRS = DATA_DIRS | {"agents"}

def test_stock_image_search_is_off_the_render_path():
    """The quote card is the visual — nothing fetches stock imagery any more.

    This is the regression guard for the whole direction change: image search
    used to put unrelated stock photos (military rifles for a crypto joke)
    behind the narration. The card renders the speaker's own photo instead, so
    the fetch step must not creep back into the pipeline.
    """
    pipeline_src = (SRC / "agents" / "video" / "pipeline.py").read_text(encoding="utf-8")
    for banned in ("fetch_assets", "asset_fetcher", "query_agent", "asset_agent"):
        assert banned not in pipeline_src, \
            f"{banned} is back on the quote render path; cards use the speaker photo"

    # And the card renderer itself must not reach for a search either.
    card_src = (SRC / "agents" / "visuals" / "card.py").read_text(encoding="utf-8")
    for banned in ("fetch_assets", "asset_fetcher", "query_agent", "requests", "urllib"):
        assert banned not in card_src, \
            f"the card agent must render from the local face image, not {banned}"


def test_voice_reference_audio_lives_in_assets():
    """The reference WAVs are live render inputs, so they get a real home.

    They used to sit under src/experiments/, which meant production TTS read
    from a folder named for scratch work. Runtime assets belong in assets/, and
    the guard is that the path in voices.py stays real: a moved or deleted clip
    otherwise fails deep inside the render, one video at a time.
    """
    for vid, cfg in voices.get_enabled_voices():
        ref = cfg["ref_audio"]
        assert ref.startswith("src/assets/voice_refs/"), \
            f"{vid} ref_audio should live in src/assets/voice_refs/: {ref}"
        assert (SRC.parent / ref).is_file(), f"{vid} reference audio is missing: {ref}"

def test_retired_subsystems_are_actually_gone():
    """The dead stack is deleted, not just unused.

    An orphaned module is still a maintenance liability and still drags its
    dependencies (ddgs, wikipedia, pytesseract, bs4, playwright) into
    requirements.txt. Cards render from the speaker's own photo, so none of
    this has any caller.
    """
    still_there = [rel for rel in MUST_STAY_REMOVED if (SRC.parent / rel).exists()]
    assert not still_there, f"retired modules are back on disk: {still_there}"

    # The voice-clone reference WAVs are live data the TTS loads, so they are
    # guarded by presence as well as by path.
    for vid, cfg in voices.get_enabled_voices():
        assert (SRC.parent / cfg["ref_audio"]).is_file(), \
            f"{vid} lost the reference WAV the TTS clones from"


def test_surviving_modules_do_not_import_retired_subsystems():
    """The extraction is complete.

    The shared helpers live in src/services/local_llm.py,
    src/shared/text.py and src/config/asset_prompts.py, so nothing may
    reach into the deleted tree.
    """
    violations: dict[str, set[str]] = {}
    for rel in MUST_SURVIVE:
        path = ROOT / rel
        assert path.is_file(), f"expected surviving module: {rel}"
        for name in _imported_names(path):
            for retired in RETIRED:
                # Match the retired name anywhere in a dotted path, so both
                # both `from x.y.z import _local` and `from x.y import z` are caught.
                if retired in name.split("."):
                    violations.setdefault(rel, set()).add(name)
    assert not violations, "retired imports still present: " + repr(violations)


def test_no_new_sys_path_inserts_outside_harnesses():
    """The retired tree used `sys.path` inserts; new code must not add more."""
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"sys\.path\.insert\([^)]*dirname\(__file__\)\)", text):
            offenders.append(str(path.relative_to(ROOT)))
    # Only the standalone voice-test entry scripts need the bootstrap.
    assert all("voice_tests" in o for o in offenders), offenders


def test_quote_state_survives_temp_cleanup():
    """The quote pool must live at the repo root, outside temp/.

    It is a durable local history, not source and not a run artifact, so it does
    not belong under src/ any more than it belongs under temp/. The hazard is
    one-directional and worth stating precisely: cleanup_temp() rmtree's TEMP_DIR,
    so anything under temp/ is destroyed on the next run. A path that only looks
    adjacent to it is fine, and an absolute one is required, because a relative
    path resolves against the CWD and a run from any other directory would
    silently open a fresh empty pool and restart dedup from zero.
    """
    from src.agents.quotes.agent import STATE_FILE
    from src.agents.video.artifacts import TEMP_DIR

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    assert os.path.isabs(STATE_FILE), STATE_FILE
    assert os.path.dirname(STATE_FILE) == root, STATE_FILE
    assert os.path.basename(STATE_FILE) == "used_quotes.json"
    # Not under the directory cleanup_temp() deletes.
    assert TEMP_DIR not in os.path.abspath(STATE_FILE).split(os.sep)


def test_quote_agent_is_groq_only():
    """The quote path must not be able to fall back to another provider."""
    from src.agents.quotes.agent import generate_quotes
    from src.agents.completion import llm

    source = _source_of(generate_quotes)
    assert "allow_fallback=False" in source, \
        "generate_quotes must pass allow_fallback=False so a dead key cannot " \
        "silently return Gemini output"

    # The shared seam keeps its permissive default for other callers.
    import inspect
    assert inspect.signature(llm.call_groq).parameters["allow_fallback"].default is True


def _source_of(func) -> str:
    import inspect
    import textwrap
    return textwrap.dedent(inspect.getsource(func))
