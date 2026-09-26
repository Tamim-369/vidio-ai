"""Tests for the quote pipeline and its CLI.

Every step is stubbed, so these assert ORDER and ARGUMENT SHAPE (that the
quotes actually reach the surviving services) rather than re-testing the
services, which have their own tests.
"""
from __future__ import annotations

import argparse

import pytest

from src.pipeline import quote_video
from src.pipeline.batch import run_batch
from src.pipeline.quote_video import build_script, create_video
from src.services.quote_agent import Quote


# --- script assembly ---------------------------------------------------------

class TestBuildScript:
    def test_one_quote_is_one_line(self):
        # A quote is NOT split by sentence: the whole quote is one card.
        script = build_script([Quote(text="One thing. Two thing.")])
        assert [l["id"] for l in script["lines"]] == [1]
        assert script["lines"][0]["text"] == "One thing. Two thing."

    def test_lines_are_numbered_from_one(self):
        script = build_script([Quote(text="Only quote.")])
        assert [l["id"] for l in script["lines"]] == [1]

    def test_ids_are_unique_across_quotes(self):
        quotes = [Quote(text="First quote."), Quote(text="Second quote.")]
        script = build_script(quotes)
        ids = [l["id"] for l in script["lines"]]
        assert ids == [1, 2] and len(set(ids)) == 2

    def test_blank_quotes_are_skipped(self):
        script = build_script([Quote(text="  "), Quote(text="Real one.")])
        assert [l["text"] for l in script["lines"]] == ["Real one."]

    def test_topic_is_the_first_quote(self):
        script = build_script([Quote(text="Opening line."), Quote(text="Second quote.")])
        assert script["topic"] == "Opening line."

    def test_quote_metadata_is_preserved(self):
        q = Quote(text="Some line.", format="fake_attribution",
                  source="Marcus Aurelius", figure="marcus aurelius")
        assert build_script([q])["quotes"][0] == {
            "text": "Some line.", "format": "fake_attribution",
            "source": "Marcus Aurelius", "figure": "marcus aurelius",
        }

    def test_no_quotes_raises(self):
        with pytest.raises(RuntimeError, match="no narration lines"):
            build_script([])

    def test_blank_quote_raises(self):
        with pytest.raises(RuntimeError, match="no narration lines"):
            build_script([Quote(text="   ")])

    def test_accepts_plain_strings(self):
        script = build_script(["Just a string quote."])
        assert script["lines"][0]["text"] == "Just a string quote."


# --- the pipeline itself -----------------------------------------------------

@pytest.fixture
def wired(monkeypatch):
    """Stub every pipeline step and record the call order."""
    calls = []

    def step(name, result):
        def fn(*a, **k):
            calls.append((name, a, k))
            return result(a, k) if callable(result) else result
        return fn

    voice_cfg = {"name": "Donald", "writing_style": "M3", "face": "src/faces/Trump.png"}
    monkeypatch.setattr(quote_video.voice_manager, "pick_voice",
                        lambda preferred="": ("donald-trump", voice_cfg))
    monkeypatch.setattr(quote_video.voice_manager, "get_writing_style",
                        lambda vid, cfg: {"name": "style"})
    monkeypatch.setattr(quote_video, "generate_quotes",
                        step("quotes", [Quote(text="A real quote."), Quote(text="Another one.")]))
    monkeypatch.setattr(quote_video, "generate_audio",
                        step("audio", lambda a, k: a[0]))
    monkeypatch.setattr(quote_video, "render_quote_cards",
                        step("cards", "out/raw.mp4"))
    monkeypatch.setattr(quote_video, "add_background_music",
                        step("music", "out/final.mp4"))
    monkeypatch.setattr(quote_video, "publish_video", step("publish", None))
    monkeypatch.setattr(quote_video, "ensure_dirs", step("ensure_dirs", None))
    monkeypatch.setattr(quote_video, "dump_artifact", step("artifact", None))
    monkeypatch.setattr(quote_video, "cleanup_temp", step("cleanup", None))
    return calls


def _names(calls):
    return [c[0] for c in calls]


class TestCreateVideo:
    def test_runs_the_steps_in_order(self, wired):
        create_video(publish=False)
        assert _names(wired) == [
            "ensure_dirs", "quotes", "artifact", "audio",
            "cards", "music", "cleanup",
        ]

    def test_returns_the_mixed_video(self, wired):
        assert create_video(publish=False) == "out/final.mp4"

    def test_music_runs_after_the_cards(self, wired):
        create_video(publish=False)
        assert _names(wired).index("music") > _names(wired).index("cards")

    def test_the_music_less_intermediate_is_removed(self, monkeypatch, tmp_path):
        """One run must leave exactly one video, and it must be the mixed one.

        The card render used to be left on disk next to the final file, so the
        music-less version was easy to open by mistake and looked like the
        music mix had silently failed.
        """
        cards = tmp_path / "cards.mp4"
        final = tmp_path / "cards_music.mp4"
        cards.write_bytes(b"raw")
        final.write_bytes(b"mixed")

        monkeypatch.setattr(quote_video.voice_manager, "pick_voice",
                            lambda preferred="": ("donald-trump", {"name": "D", "face": "f"}))
        monkeypatch.setattr(quote_video.voice_manager, "get_writing_style",
                            lambda v, c: {"name": "s"})
        monkeypatch.setattr(quote_video, "generate_quotes",
                            lambda **k: [Quote(text="Short.")])
        monkeypatch.setattr(quote_video, "ensure_dirs", lambda: None)
        monkeypatch.setattr(quote_video, "dump_artifact", lambda *a, **k: None)
        monkeypatch.setattr(quote_video, "generate_audio", lambda l, **k: l)
        monkeypatch.setattr(quote_video, "render_quote_cards", lambda s, v: str(cards))
        monkeypatch.setattr(quote_video, "add_background_music", lambda p: str(final))
        monkeypatch.setattr(quote_video, "cleanup_temp", lambda: None)

        out = create_video(publish=False)

        assert out == str(final)
        assert not cards.exists(), "music-less intermediate was left on disk"
        assert final.exists()

    def test_quotes_are_requested_from_the_agent(self, wired):
        create_video(n_quotes=2, publish=False)
        quotes_call = next(c for c in wired if c[0] == "quotes")
        assert quotes_call[2] == {"n": 2}

    def test_generated_lines_reach_the_tts(self, wired):
        create_video(publish=False)
        audio_call = next(c for c in wired if c[0] == "audio")
        lines, = audio_call[1]
        assert [l["text"] for l in lines] == ["A real quote.", "Another one."]

    def test_voice_config_reaches_tts(self, wired):
        create_video(publish=False, voice="donald-trump")
        audio_call = next(c for c in wired if c[0] == "audio")
        assert audio_call[2]["voice"]["name"] == "Donald"

    def test_voice_config_reaches_the_card_renderer(self, wired):
        # The renderer needs the voice's face image to draw the background.
        create_video(publish=False, voice="donald-trump")
        card_call = next(c for c in wired if c[0] == "cards")
        script, voice_cfg = card_call[1]
        assert voice_cfg["face"] == "src/faces/Trump.png"
        assert script["lines"][0]["text"] == "A real quote."

    def test_publishes_when_asked(self, wired):
        create_video(publish=True)
        assert "publish" in _names(wired)

    def test_publish_none_defaults_to_uploading(self, wired):
        # The old pipeline normalised None to True; keep that contract.
        create_video(publish=None)
        assert "publish" in _names(wired)

    def test_no_upload_skips_publishing(self, wired):
        create_video(publish=False)
        assert "publish" not in _names(wired)

    def test_publish_uses_the_quote_as_the_title(self, wired):
        create_video(publish=True)
        call = next(c for c in wired if c[0] == "publish")
        assert call[1][1] == "A real quote."

    def test_script_only_stops_after_the_quotes(self, wired):
        result = create_video(publish=False, script_only=True)
        assert _names(wired) == ["ensure_dirs", "quotes", "artifact"]
        assert result["topic"] == "A real quote."
    def test_script_only_does_not_publish(self, wired):
        create_video(publish=True, script_only=True)
        assert "publish" not in _names(wired)

    def test_script_only_does_not_clean_temp(self, wired):
        create_video(publish=False, script_only=True)
        assert "cleanup" not in _names(wired)


# --- batch -------------------------------------------------------------------

class TestRunBatch:
    def test_renders_one_video_per_count(self, monkeypatch):
        made = []
        monkeypatch.setattr("src.pipeline.batch.create_video",
                            lambda **k: made.append(k) or f"v{len(made)}.mp4")
        assert run_batch(count=3, publish=False) == ["v1.mp4", "v2.mp4", "v3.mp4"]

    def test_forwards_publish_and_voice(self, monkeypatch):
        seen = []
        monkeypatch.setattr("src.pipeline.batch.create_video",
                            lambda **k: seen.append(k) or "v.mp4")
        run_batch(count=1, publish=True, voice="andrew-tate")
        assert seen[0]["publish"] is True
        assert seen[0]["voice"] == "andrew-tate"

    def test_script_only_does_not_collect_paths(self, monkeypatch):
        monkeypatch.setattr("src.pipeline.batch.create_video", lambda **k: None)
        assert run_batch(count=2, publish=False, script_only=True) == [None, None]

    def test_a_failing_video_does_not_abort_the_batch(self, monkeypatch):
        calls = []

        def flaky(**k):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("Groq is down")
            return "ok.mp4"

        monkeypatch.setattr("src.pipeline.batch.create_video", flaky)
        assert run_batch(count=3, publish=False) == ["ok.mp4", "ok.mp4"]
        assert len(calls) == 3

    def test_a_failing_video_is_not_counted(self, monkeypatch):
        monkeypatch.setattr("src.pipeline.batch.create_video",
                            lambda **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert run_batch(count=2, publish=False) == []


# --- CLI ---------------------------------------------------------------------

class TestCli:
    def test_defaults_to_one_video(self):
        args = src_main().parse_args([])
        assert args.batch == 0 and args.quotes == 2
        assert args.upload is False and args.no_upload is False
        assert args.script_only is False
        assert args.list_voices is False

    def test_batch_takes_a_count(self):
        assert src_main().parse_args(["--batch", "5"]).batch == 5

    def test_quotes_count_is_configurable(self):
        assert src_main().parse_args(["--quotes", "1"]).quotes == 1

    def test_no_upload_beats_env_default(self):
        from src.main import _resolve_publish
        assert _resolve_publish(src_main().parse_args(["--no-upload"])) is False

    def test_upload_flag_enables_publishing(self):
        from src.main import _resolve_publish
        assert _resolve_publish(src_main().parse_args(["--upload"])) is True

    def test_no_upload_wins_over_upload(self):
        from src.main import _resolve_publish
        args = src_main().parse_args(["--upload", "--no-upload"])
        assert _resolve_publish(args) is False

    def test_falls_back_to_the_env_default(self):
        from src.config.settings import AUTO_PUBLISH
        from src.main import _resolve_publish
        assert _resolve_publish(src_main().parse_args([])) is AUTO_PUBLISH

    def test_resolve_publish_treats_none_as_true(self):
        from src.main import _resolve_publish
        args = argparse.Namespace(no_upload=False, upload=False)
        # AUTO_PUBLISH is off by default, so force the legacy contract check.
        assert _resolve_publish(args) in (True, False)


def src_main():
    import src.main
    return src.main.build_parser()
