"""Characterization tests for filesystem helpers and voice selection.

``file_helpers`` and ``voice_manager`` sit on the hot path of every run, so their
slugs, JSON round-trips, and round-robin order are behaviour worth pinning.
"""
from __future__ import annotations

import json
import os

import pytest

from src.agents.voice_cast import agent as voice_manager
from src.agents.video import artifacts as file_helpers


class TestOutputPath:
    @pytest.mark.parametrize("topic,expected", [
        ("The Great Wall", "the_great_wall"),
        ("WWII: A Short History!", "wwii_a_short_history"),
        ("  spaced  out  ", "spaced_out"),
        ("Mixed CASE topic", "mixed_case_topic"),
    ])
    def test_slug_generation(self, topic, expected):
        # Paths are derived from topics, so slugs must stay filesystem-safe.
        from src.agents.visuals.card import output_path
        stem = os.path.basename(output_path(topic))
        assert stem.endswith(f"_{expected}.mp4"), stem

    def test_long_topic_is_capped(self):
        from src.agents.visuals.card import output_path
        stem = os.path.basename(output_path("word " * 200))[: -len(".mp4")]
        assert len(stem) <= 60

    def test_path_is_under_output_dir(self):
        from src.agents.visuals.card import OUTPUT_DIR, output_path
        assert output_path("x").startswith(OUTPUT_DIR)

    def test_two_videos_on_one_topic_get_different_files(self, tmp_path,
                                                         monkeypatch):
        # The reason the name carries a timestamp. A video is one character
        # saying one quote, so a batch has only as many topics as there are
        # characters -- naming on the topic alone made every video from the same
        # character overwrite the last one.
        import src.agents.visuals.card as card
        monkeypatch.setattr(card, "OUTPUT_DIR", str(tmp_path))
        first = card.output_path("War and military strategy")
        (tmp_path / os.path.basename(first)).write_bytes(b"x")
        second = card.output_path("War and military strategy")
        assert first != second

    def test_names_sort_chronologically(self, tmp_path, monkeypatch):
        import src.agents.visuals.card as card
        monkeypatch.setattr(card, "OUTPUT_DIR", str(tmp_path))
        stem = os.path.basename(card.output_path("War"))
        assert stem[:8].isdigit() and stem[8] == "_"


class TestDumpArtifact:
    def test_dump_artifact_writes_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = file_helpers.dump_artifact("topics", {"a": 1}, topic="Some Topic")
        assert path.endswith(".json")
        assert json.load(open(path)) == {"a": 1}

    def test_dump_artifact_writes_text_verbatim(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = file_helpers.dump_artifact("script", "line one\nline two", topic="T")
        assert path.endswith(".txt")
        assert open(path).read() == "line one\nline two"

    def test_artifact_names_do_not_clobber(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a = file_helpers.dump_artifact("s", "x")
        b = file_helpers.dump_artifact("s", "x")
        assert a != b


class TestEnsureDirs:
    def test_creates_expected_tree(self, tmp_path, monkeypatch):
        out = str(tmp_path / "out")
        temp = str(tmp_path / "tmp")
        monkeypatch.setattr(file_helpers, "OUTPUT_DIR", out)
        monkeypatch.setattr(file_helpers, "TEMP_DIR", temp)
        file_helpers.ensure_dirs()
        assert os.path.isdir(out)
        assert os.path.isdir(f"{temp}/assets")
        assert os.path.isdir(f"{temp}/audio")

    def test_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_helpers, "TEMP_DIR", str(tmp_path / "tmp"))
        file_helpers.ensure_dirs()
        file_helpers.ensure_dirs()  # must not raise


class TestPickVoice:
    """Smoke coverage only.

    Rotation, persistence and subject pairing are covered properly in
    test_voice_rotation.py. These stay here as a check that the voice manager is
    wired and exportable, and every one of them passes an explicit tmp path so
    the suite can never advance the real voice_rotation.json cursor.
    """

    def test_honours_preferred(self, tmp_path):
        vid, _ = voice_manager.pick_voice(preferred="andrew-tate",
                                          path=str(tmp_path / "rot.json"))
        assert vid == "andrew-tate"

    def test_never_returns_disabled_voice(self, tmp_path):
        path = str(tmp_path / "rot.json")
        for _ in range(9):
            vid, _ = voice_manager.pick_voice(path=path)
            assert vid != "narrator"


class TestGetWritingStyle:
    def test_resolves_by_voice_entry(self):
        style = voice_manager.get_writing_style("andrew-tate", {"writing_style": "arnold"})
        assert style is not None
        assert "persona" in style
