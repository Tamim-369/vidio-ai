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

from src.agents.voice_cast import voices

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
    "src/agents/video/assemble.py",
    "src/agents/video/artifacts.py",
    "src/agents/visuals/layout.py",
    "src/agents/voiceover/pauses.py",
    "src/agents/voiceover/models.py",
    "src/agents/voiceover/workers.py",
    "src/agents/publish/metadata.py",
    "src/main.py",
    "src/experiments/voice_tests/voice_test_utils.py",
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
    from src.agents.voice_cast.voices import get_enabled_voices

    for voice_id, voice in get_enabled_voices():
        face = voice.get("face")
        assert face, f"enabled voice {voice_id!r} has no 'face' image"
        assert (ROOT / face).is_file(), f"voice {voice_id!r} face not on disk: {face}"
        assert voice.get("quote_authors"), \
            f"enabled voice {voice_id!r} has no fake quote authors to credit"


# Directories under src/ that are not part of the code layout. `faces`, `music`
# and `state` are runtime data; `tests` and `experiments` are separate trees.
DATA_DIRS = {"faces", "music", "state", "tests", "experiments", "__pycache__"}
ALLOWED_DIRS = DATA_DIRS | {"agents"}


def test_no_directory_pileup_at_the_src_root():
    """The layout is agents/ + flat modules. Every extra layer has to be earned.

    A config/, services/ or shared/ layer used to exist to hold values and
    helpers that each had exactly one caller. Colocating them with their caller
    is why those are gone; this pins the shape so they cannot quietly return.
    """
    found = {p.name for p in SRC.iterdir() if p.is_dir()} - DATA_DIRS
    assert found == {"agents"}, \
        f"src/ should hold only agents/ plus flat modules, found extra: {sorted(found)}"

    # The agent packages themselves stay flat: task folder + its modules only.
    for folder in (SRC / "agents").iterdir():
        if not folder.is_dir() or folder.name == "__pycache__":
            continue
        nested = [p.name for p in folder.iterdir() if p.is_dir() and p.name != "__pycache__"]
        assert not nested, f"agents/{folder.name}/ should be flat, found subdirs: {nested}"

    # No package-per-concern: these are modules, not directories.
    for name in ("config", "services", "shared", "utils", "pipeline"):
        assert not (SRC / name).is_dir(), f"src/{name}/ is not a layer any more"

    # And the flat layer at the root is an explicit, small set. A new module
    # here is a new layer, so it has to be added here on purpose.
    allowed = {"__init__.py", "main.py"}
    modules = {p.name for p in SRC.glob("*.py")} - allowed
    assert not modules, f"unexpected modules at the src root: {sorted(modules)}"


def test_every_agent_is_a_documented_self_contained_task():
    """One folder per task, and the folder is the only public door.

    A folder per task, an __init__ that re-exports the entry point behind a
    one-line docstring, and nothing importing another agent's internals.
    """
    agents = SRC / "agents"
    folders = sorted(p for p in agents.iterdir() if p.is_dir() and p.name != "__pycache__")
    assert folders, "src/agents/ has no agent folders"

    for folder in folders:
        init = folder / "__init__.py"
        assert init.is_file(), f"{folder.name} has no __init__.py to export its task"

        doc = ast.get_docstring(ast.parse(init.read_text(encoding="utf-8"))) or ""
        assert doc, f"{folder.name}/__init__.py needs a one-line docstring"
        assert len(doc.splitlines()) == 1, \
            f"{folder.name}/__init__.py docstring should be one line, got {len(doc.splitlines())}"

        # The package must actually export its entry point, and every name it
        # advertises must resolve -- an __all__ naming a missing symbol would
        # otherwise pass a non-empty check while breaking every caller.
        module = __import__(f"src.agents.{folder.name}", fromlist=["x"])
        exported = list(getattr(module, "__all__", []))
        assert exported, f"{folder.name}/__init__.py re-exports nothing"
        missing = [n for n in exported if not hasattr(module, n)]
        assert not missing, \
            f"{folder.name}/__init__.py advertises names it does not define: {missing}"

    # Naming: the folder is the task, so it should not be named after a
    # technology. "provider" / "util" / "helper" describe a mechanism, not a task.
    banned = {"providers", "utils", "helpers", "common", "misc", "lib"}
    assert not ({f.name for f in folders} & banned), \
        "agent folders are named after tasks, not mechanisms"


def test_stock_image_search_is_off_the_render_path():
    """The quote card is the visual — nothing fetches stock imagery any more.

    This is the regression guard for the whole direction change: image search
    used to put unrelated stock photos (military rifles for a crypto joke)
    behind the narration. The card renders the speaker's own photo instead, so
    the fetch step must not creep back into the pipeline.
    """
    pipeline_src = (SRC / "agents" / "video" / "assemble.py").read_text(encoding="utf-8")
    for banned in ("fetch_assets", "asset_fetcher", "query_agent", "asset_agent"):
        assert banned not in pipeline_src, \
            f"{banned} is back on the quote render path; cards use the speaker photo"

    # And the card renderer itself must not reach for a search either.
    card_src = (SRC / "agents" / "visuals" / "card.py").read_text(encoding="utf-8")
    for banned in ("fetch_assets", "asset_fetcher", "query_agent", "requests", "urllib"):
        assert banned not in card_src, \
            f"the card agent must render from the local face image, not {banned}"


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
    # Checked against real imports, not raw text: a docstring may legitimately
    # mention src/experiments/ when explaining where something came from.
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts or "experiments" in path.parts:
            continue
        bad = [n for n in _imported_names(path) if n.split(".")[1:2] == ["experiments"]]
        assert not bad, f"{path.name} must not reach into src/experiments/: {bad}"


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


def test_quote_state_stays_inside_src():
    """The quote pool must not be written outside src/.

    Root-level state files are not in scope for this project, and a pool at the
    repo root would also be wiped by the temp cleanup.
    """
    from src.agents.quotes.agent import STATE_FILE

    assert STATE_FILE.startswith("src/"), STATE_FILE


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
