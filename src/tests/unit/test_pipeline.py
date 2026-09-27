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
    def test_the_quote_is_one_line(self):
        # A quote is NOT split by sentence: the whole quote is one card.
        script = build_script(TRUMP, Quote(text="One thing. Two thing."))
        assert [l["id"] for l in script["lines"]] == [1]
        assert script["lines"][0]["text"] == "One thing. Two thing."

    def test_the_topic_is_the_characters_subject(self):
        # The subject names the file and titles the upload; a random joke as the
        # topic produced titles that read like the joke itself.
        script = build_script(TRUMP, Quote(text="Opening line."))
        assert script["topic"] == "War and military strategy"

    def test_the_line_carries_the_character_and_its_config(self):
        # TTS and the card renderer both read the speaker from the line, so it
        # has to be there without either of them looking the id up again.
        script = build_script(TRUMP, Quote(text="A line."))
        line = script["lines"][0]
        assert line["character"] == TRUMP
        assert line["voice"]["pseudonym"] == "Don Tzu"

    def test_exactly_one_character_and_one_quote(self):
        script = build_script(TRUMP, Quote(text="A line."))
        assert script["characters"] == [TRUMP]
        assert script["quotes"] == [{"text": "A line."}]

    def test_quote_text_is_preserved(self):
        assert build_script(TRUMP, Quote(text="Some line."))["quotes"] == \
            [{"text": "Some line."}]

    def test_blank_quote_raises(self):
        with pytest.raises(RuntimeError, match="empty quote"):
            build_script(TRUMP, Quote(text="   "))

    def test_accepts_plain_strings(self):
        assert build_script(TRUMP, "Just a string quote.")["lines"][0]["text"] == \
            "Just a string quote."


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

    def test_publish_uses_the_subject_as_the_topic(self, wired):
        create_video(publish=True)
        args = next(a for n, a, _ in wired if n == "publish")
        assert args[1] == "War and military strategy"

    def test_the_title_is_the_pseudonym_and_the_quote(self, wired):
        create_video(publish=False)
        assert _args(wired, "cards")[0]["title"] == "Don Tzu: A real quote."

    def test_script_only_stops_after_the_quote(self, wired):
        create_video(publish=True, script_only=True)
        assert _names(wired) == ["ensure_dirs", "quotes", "artifact"]

    def test_script_only_does_not_publish(self, wired):
        create_video(publish=True, script_only=True)
        assert "publish" not in _names(wired)

    def test_script_only_does_not_clean_temp(self, wired):
        create_video(publish=True, script_only=True)
        assert "cleanup" not in _names(wired)

    def test_script_only_explains_dropped_candidates(self, wired):
        # Otherwise a filtered candidate looks like the model being stingy.
        create_video(script_only=True)
        assert _kw(wired, "quotes")["explain"] is True


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

    def test_a_failing_video_does_not_abort_the_batch(self, monkeypatch,
                                                     no_side_effects):
        def boom(**k):
            if len(no_side_effects) == 1:
                no_side_effects.append(k)
                raise RuntimeError("tts exploded")
            no_side_effects.append(k)
            return "out/ok.mp4"

        monkeypatch.setattr(video, "create_video", boom)
        out = run_batch(count=3, publish=False)
        assert len(out) == 2, "one failure must not cost the other videos"

    def test_a_failing_video_is_not_counted(self, monkeypatch):
        monkeypatch.setattr(video, "create_video",
                            lambda **k: (_ for _ in ()).throw(RuntimeError("x")))
        assert run_batch(count=2, publish=False) == []

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

    def test_defaults_to_one_video(self, captured):
        import src.main as main
        main.main.__wrapped__ if False else None
        sys_argv = None
        # main() reads sys.argv; exercise the parser directly for the default.
        args = self._parse([])
        assert args.batch == 0

    def test_batch_takes_a_count(self):
        assert self._parse(["--batch", "10"]).batch == 10

    def test_voice_defaults_to_rotating(self):
        assert self._parse([]).voice == ""

    def test_voice_can_be_forced(self):
        assert self._parse(["--voice", TRUMP]).voice == TRUMP

    def test_script_only_is_off_by_default(self):
        assert self._parse([]).script_only is False

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

    def test_no_upload_keeps_it_local(self, monkeypatch):
        assert self._run(["--no-upload"], monkeypatch)["publish"] is False

    def test_upload_flag_enables_publishing(self, monkeypatch):
        assert self._run(["--upload"], monkeypatch)["publish"] is True

    def test_no_upload_wins_over_upload(self, monkeypatch):
        assert self._run(["--upload", "--no-upload"], monkeypatch)["publish"] is False

    def test_falls_back_to_the_env_default(self, monkeypatch):
        monkeypatch.setenv("AUTO_PUBLISH", "1")
        assert self._run([], monkeypatch)["publish"] is True

    def test_env_default_is_off_without_the_flag(self, monkeypatch):
        monkeypatch.delenv("AUTO_PUBLISH", raising=False)
        assert self._run([], monkeypatch)["publish"] is False
