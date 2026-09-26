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
import re
from pathlib import Path

from src.config import voices

# src/tests/unit/test_architecture.py -> repo root
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

# Pre-cleanup copies recovered from the old repo-root voice_tests/, kept purely as
# a record. They still import the long-deleted research modules, so they are
# exempt from the guards below -- but nothing live may import them.
ARCHIVE_DIR = SRC / "experiments" / "voice_tests" / "_superseded_repo_root"

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
    "src/services/quote_agent.py",
    "src/services/quote_card.py",
    "src/services/music.py",
    "src/services/video_assembler.py",
    "src/services/captions.py",
    "src/services/tts.py",
    "src/services/tts_dsp.py",
    "src/services/tts_text.py",
    "src/services/voice_manager.py",
    "src/services/youtube_upload.py",
    "src/services/llm.py",
    "src/config/quote_prompt.py",
    "src/config/voices.py",
    "src/config/settings.py",
    "src/config/prompt.py",
    "src/config/writing_styles.py",
    "src/pipeline/quote_video.py",
    "src/pipeline/batch.py",
    "src/utils/file_helpers.py",
    "src/utils/text_helpers.py",
    "src/experiments/voice_tests/voice_test_utils.py",
]

# The retired subsystems. Kept as a regression guard in the other direction:
# the direction change was that cards render from the speaker's own photo, so
# the image-search stack and the manual voice-clone scripts have no place in
# this repo and should not creep back in.
MUST_STAY_REMOVED = [
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


def test_retired_subsystems_are_gone():
    """The removal is complete and must stay complete."""
    still_there = [rel for rel in REMOVED_PATHS if (ROOT / rel).exists()]
    assert not still_there, f"retired paths came back: {still_there}"


def test_nothing_imports_the_retired_subsystems():
    """No file in src/ may reference a retired module any more.

    Stricter than the pre-removal guard: previously the retired tree was allowed
    to import itself, and the live harness was allow-listed. Now that the tree is
    deleted, any hit is either a live break or a resurrection attempt.
    """
    offenders: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        if ARCHIVE_DIR in path.parents:  # inert historical record, not live code
            continue
        rel = str(path.relative_to(ROOT))
        hits = {n for n in _imported_names(path)
                for r in RETIRED if r in n.split(".")}
        if hits:
            offenders[rel] = hits
    assert not offenders, f"imports of retired code remain: {offenders}"


def test_nothing_live_imports_the_superseded_archive():
    """The archive exists as a record only; it must stay inert.

    Otherwise a future refactor could start depending on a pre-cleanup copy and
    quietly resurrect the retired research/asset code paths.
    """
    live = [p for p in sorted(SRC.rglob("*.py"))
            if ARCHIVE_DIR not in p.parents and "__pycache__" not in p.parts]
    offenders = [str(p.relative_to(ROOT)) for p in live
                 if any("_superseded_repo_root" in n
                        for n in _imported_names(p))]
    assert not offenders, f"live code imports the archive: {offenders}"


def test_surviving_modules_all_exist():
    """The quote pipeline's service set is intact."""
    missing = [rel for rel in MUST_SURVIVE if not (ROOT / rel).is_file()]
    assert not missing, f"surviving modules went missing: {missing}"


def test_speaker_faces_exist():
    """Every card background must be present, or rendering fails at runtime."""
    missing = [rel for rel in REQUIRED_FACES if not (ROOT / rel).is_file()]
    assert not missing, f"speaker face images went missing: {missing}"


def test_every_enabled_voice_has_a_face_and_authors():
    """A voice without a face or a fake author cannot produce a card."""
    from src.config.voices import get_enabled_voices

    for voice_id, voice in get_enabled_voices():
        face = voice.get("face")
        assert face, f"enabled voice {voice_id!r} has no 'face' image"
        assert (ROOT / face).is_file(), f"voice {voice_id!r} face not on disk: {face}"
        assert voice.get("quote_authors"), \
            f"enabled voice {voice_id!r} has no fake quote authors to credit"


def test_stock_image_search_is_off_the_render_path():
    """The quote card is the visual — nothing fetches stock imagery any more.

    This is the regression guard for the whole direction change: image search
    used to put unrelated stock photos (military rifles for a crypto joke)
    behind the narration. The card renders the speaker's own photo instead, so
    the fetch step must not creep back into the pipeline.
    """
    pipeline_src = (SRC / "pipeline" / "quote_video.py").read_text(encoding="utf-8")
    for banned in ("fetch_assets", "asset_fetcher", "query_agent", "asset_agent"):
        assert banned not in pipeline_src, \
            f"{banned} is back on the quote render path; cards use the speaker photo"

    # And the card renderer itself must not reach for a search either.
    card_src = (SRC / "services" / "quote_card.py").read_text(encoding="utf-8")
    for banned in ("fetch_assets", "asset_fetcher", "query_agent", "requests", "urllib"):
        assert banned not in card_src, \
            f"quote_card must render from the local face image, not {banned}"


def test_experiments_are_kept_under_one_directory():
    """src/experiments/ is the home for voice work; the old roots stay empty.

    The voice-clone reference WAVs are live render inputs, so they must not be
    swept up by a cleanup pass. Both trees were consolidated here so they are
    visibly quarantined from the pipeline, and so a future prune skips them.
    """
    for old in ("voice_tests", "voices_to_clone"):
        assert not (SRC / old).exists(), \
            f"src/{old} still exists; both trees belong under src/experiments/"

    experiments = SRC / "experiments"
    assert (experiments / "voices_to_clone").is_dir()
    assert (experiments / "voice_tests").is_dir()
    assert (experiments / "voices_to_clone" / "candidates").is_dir()

    # voices.py points here; if the layout moves, that path goes stale.
    for voice in voices.get_enabled_voices():
        vid, cfg = voice
        assert cfg["ref_audio"].startswith("src/experiments/voices_to_clone/"), \
            f"{vid} ref_audio must live under src/experiments/: {cfg['ref_audio']}"

    # And nothing on the render path may import from the experiments tree.
    for path in list((SRC / "services").rglob("*.py")) + list((SRC / "pipeline").rglob("*.py")):
        assert "experiments" not in path.read_text(encoding="utf-8"), \
            f"{path.name} must not reach into src/experiments/"


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
    # guarded by presence. The preparation/QA scripts beside them are kept
    # deliberately (they live under src/experiments/ and must not be pruned).
    for voice_dir in (SRC / "experiments" / "voices_to_clone" / "candidates").iterdir():
        if voice_dir.is_dir():
            assert (voice_dir / f"{voice_dir.name}_ref.wav").is_file(), \
                f"{voice_dir.name} lost the reference WAV the TTS clones from"


def test_surviving_modules_do_not_import_retired_subsystems():
    """The extraction is complete.

    The shared helpers live in src/services/local_llm.py,
    src/utils/text_helpers.py and src/config/asset_prompts.py, so nothing may
    reach into the deleted tree.
    """
    violations: dict[str, set[str]] = {}
    for rel in MUST_SURVIVE:
        path = ROOT / rel
        assert path.is_file(), f"expected surviving module: {rel}"
        for name in _imported_names(path):
            for retired in RETIRED:
                # Match the retired name anywhere in a dotted path, so both
                # `from src.services.script_lab.llm import _local` and
                # `from src.services import script_lab` are caught.
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


def test_quote_state_stays_inside_src():
    """The quote pool must not be written outside src/.

    Root-level state files are not in scope for this project, and a pool at the
    repo root would also be wiped by the temp cleanup.
    """
    from src.services.quote_agent import STATE_FILE

    assert STATE_FILE.startswith("src/"), STATE_FILE


def test_quote_agent_is_groq_only():
    """The quote path must not be able to fall back to another provider."""
    from src.services.quote_agent import generate_quotes
    from src.services import llm

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
