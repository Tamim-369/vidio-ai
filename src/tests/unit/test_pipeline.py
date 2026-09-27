"""Tests for the video pipeline and its CLI.

Every step is stubbed, so these assert ORDER and ARGUMENT SHAPE (that the quote
actually reaches the surviving services) rather than re-testing the services,
which have their own tests.
"""
from __future__ import annotations

import pytest

from src.agents.video import pipeline as video
from src.agents.video.pipeline import build_script, create_video, run_batch
from src.agents.quotes.agent import Quote

TRUMP = "donald-trump"


# --- the one-line script ------------------------------------------------------

class TestBuildScript:
    def test_blank_quote_raises(self):
        with pytest.raises(RuntimeError, match="empty quote"):
            build_script(TRUMP, Quote(text="   "))

# --- the pipeline itself ------------------------------------------------------

@pytest.fixture
def wired(monkeypatch):
    """Stub every pipeline step and record the call order."""
    calls = []

    def step(name, result):
        def fn(*a, **k):
            calls.append((name, a, k))
            return result(a, k) if callable(result) else result
        return fn

    voice_cfg = {"name": "Donald", "pseudonym": "Don Tzu",
                 "writing_style": "M3", "face": "src/faces/Trump.png",
                 "subject": "War and military strategy"}
    monkeypatch.setattr(video, "pick_voice",
                        lambda preferred="": (TRUMP, voice_cfg))
    monkeypatch.setattr(video, "get_writing_style",
                        lambda vid, cfg: {"name": "style"})
    monkeypatch.setattr(video, "generate_quotes",
                        step("quotes", [Quote(text="A real quote.")]))
    monkeypatch.setattr(video, "generate_audio",
                        step("audio", lambda a, k: a[0]))
    monkeypatch.setattr(video, "render_quote_cards", step("cards", "out/raw.mp4"))
    monkeypatch.setattr(video, "add_background_music", step("music", "out/final.mp4"))
    monkeypatch.setattr(video, "publish_video", step("publish", None))
    monkeypatch.setattr(video, "ensure_dirs", step("ensure_dirs", None))
    monkeypatch.setattr(video, "dump_artifact", step("artifact", None))
    monkeypatch.setattr(video, "cleanup_temp", step("cleanup", None))
    return calls


def _names(calls):
    return [c[0] for c in calls]


def _kw(calls, step_name):
    return next(k for n, _, k in calls if n == step_name)


def _args(calls, step_name):
    return next(a for n, a, _ in calls if n == step_name)


class TestCreateVideo:
    def test_runs_the_steps_in_order(self, wired):
        create_video(publish=False)
        assert _names(wired) == ["ensure_dirs", "quotes", "artifact", "audio",
                                "cards", "music", "cleanup"]

    def test_returns_the_final_path(self, wired):
        assert create_video(publish=False) == "out/final.mp4"

    def test_music_runs_after_the_cards(self, wired):
        create_video(publish=False)
        assert _names(wired).index("music") > _names(wired).index("cards")

    def test_asks_for_exactly_one_quote(self, wired):
        # One character, one quote. The line count is not a knob any more.
        create_video(publish=False)
        assert _kw(wired, "quotes")["n"] == 1

    def test_the_quote_is_requested_on_that_characters_subject(self, wired):
        create_video(character=TRUMP, publish=False)
        kw = _kw(wired, "quotes")
        assert kw["character"] == TRUMP
        assert kw["subject"] == "War and military strategy"

    def test_the_script_line_reaches_tts(self, wired):
        create_video(publish=False)
        assert _args(wired, "audio")[0][0]["text"] == "A real quote."

    def test_the_voice_config_reaches_the_card_renderer(self, wired):
        create_video(publish=False)
        script = _args(wired, "cards")[0]
        assert script["lines"][0]["voice"]["pseudonym"] == "Don Tzu"

    def test_publishes_when_asked(self, wired):
        create_video(publish=True)
        assert "publish" in _names(wired)

    def test_publish_none_defaults_to_uploading(self, wired):
        create_video(publish=None)
        assert "publish" in _names(wired)

    def test_no_upload_skips_publishing(self, wired):
        create_video(publish=False)
        assert "publish" not in _names(wired)

    def test_script_only_stops_after_the_quote(self, wired):
        create_video(publish=True, script_only=True)
        assert _names(wired) == ["ensure_dirs", "quotes", "artifact"]

    def test_script_only_does_not_publish(self, wired):
        create_video(publish=True, script_only=True)
        assert "publish" not in _names(wired)

class TestRunBatch:
    @pytest.fixture(autouse=True)
    def no_side_effects(self, monkeypatch):
        made = []
        monkeypatch.setattr(video, "create_video",
                            lambda **k: made.append(k) or f"out/{len(made)}.mp4")
        return made

    def test_makes_one_video_per_count(self, no_side_effects):
        out = run_batch(count=4, publish=False)
        assert len(no_side_effects) == 4 and len(out) == 4

    def test_forwards_publish(self, no_side_effects):
        run_batch(count=2, publish=True)
        assert all(c["publish"] is True for c in no_side_effects)

    def test_forwards_script_only(self, no_side_effects):
        run_batch(count=2, publish=False, script_only=True)
        assert all(c["script_only"] is True for c in no_side_effects)

    def test_passes_no_voice_by_default(self, no_side_effects):
        # Empty means "rotate", which is the batch default.
        run_batch(count=3, publish=False)
        assert all(c["character"] == "" for c in no_side_effects)

    def test_forwards_a_forced_voice(self, no_side_effects):
        run_batch(count=2, publish=False, voice=TRUMP)
        assert all(c["character"] == TRUMP for c in no_side_effects)

    def test_a_video_that_fails_once_is_retried_and_made(self, monkeypatch):
        """One flaky TTS call must not cost the slot.

        A video is one quote, so a failure anywhere in the build used to end it.
        The retry re-runs the build, which also draws a fresh quote.
        """
        calls = []

        def boom(**k):
            calls.append(k)
            if len(calls) == 1:
                raise RuntimeError("tts exploded")
            return "out/ok.mp4"

        monkeypatch.setattr(video, "create_video", boom)
        monkeypatch.setattr(video, "cleanup_temp", lambda: None)
        assert run_batch(count=1, publish=False) == ["out/ok.mp4"]
        assert len(calls) == 2, "the failed attempt should have been retried"

    def test_a_failing_video_does_not_abort_the_batch(self, monkeypatch,
                                                     no_side_effects):
        def boom(**k):
            no_side_effects.append(k)
            # video 1 exhausts both attempts; 2 and 3 are fine
            if len(no_side_effects) <= 2:
                raise RuntimeError("tts exploded")
            return "out/ok.mp4"

        monkeypatch.setattr(video, "create_video", boom)
        monkeypatch.setattr(video, "cleanup_temp", lambda: None)
        out = run_batch(count=3, publish=False)
        assert len(out) == 2, "one failure must not cost the other videos"

    def test_a_failing_video_is_not_counted(self, monkeypatch):
        calls = []

        def boom(**k):
            calls.append(k)
            raise RuntimeError("tts exploded")

        monkeypatch.setattr(video, "create_video", boom)
        monkeypatch.setattr(video, "cleanup_temp", lambda: None)
        assert run_batch(count=2, publish=False) == []
        assert len(calls) == 4, "two videos, two attempts each"

    def test_a_failed_attempt_clears_the_previous_partial_render(self, monkeypatch):
        # A failed build can leave a half-written wav behind, and the retry would
        # otherwise render on top of it.
        cleaned = []
        monkeypatch.setattr(video, "create_video",
                            lambda **k: (_ for _ in ()).throw(RuntimeError("x")))
        monkeypatch.setattr(video, "cleanup_temp", lambda: cleaned.append(1))
        run_batch(count=1, publish=False)
        assert len(cleaned) == 2, "temp should be cleared before each retry"

    def test_attempts_of_one_disables_the_retry(self, monkeypatch):
        calls = []
        monkeypatch.setattr(video, "create_video",
                            lambda **k: calls.append(k) or (_ for _ in ()).throw(
                                RuntimeError("x")))
        monkeypatch.setattr(video, "cleanup_temp", lambda: None)
        assert run_batch(count=1, publish=False, attempts=1) == []
        assert len(calls) == 1

    def test_an_unknown_voice_is_refused(self):
        with pytest.raises(ValueError, match="unknown or disabled voice"):
            run_batch(count=1, publish=False, voice="nobody")

    def test_zero_makes_nothing(self, no_side_effects):
        assert run_batch(count=0, publish=False) == []


class TestCli:
    """The CLI is four flags; the only question it asks is how many."""

    @pytest.fixture(autouse=True)
    def captured(self, monkeypatch):
        seen = {}
        monkeypatch.setattr("src.main.run_batch",
                            lambda **k: seen.update(batch=k))
        monkeypatch.setattr("src.main.create_video",
                            lambda **k: seen.update(single=k))
        return seen

    def _parse(self, argv):
        import src.main as main
        return main.build_parser().parse_args(argv)

    def test_the_old_knobs_are_gone(self):
        # One character, one quote, one video: nothing to configure.
        for flag in ("--format", "--quotes", "--seed"):
            with pytest.raises(SystemExit):
                self._parse([flag, "1"])


class TestUploadFlag:
    def _run(self, argv, monkeypatch):
        import importlib

        import src.main as main
        importlib.reload(main)
        seen = {}
        monkeypatch.setattr(main, "create_video", lambda **k: seen.update(k))
        monkeypatch.setattr("sys.argv", ["main.py"] + argv)
        main.main()
        return seen

    def test_no_upload_wins_over_upload(self, monkeypatch):
        assert self._run(["--upload", "--no-upload"], monkeypatch)["publish"] is False

