"""Tests for voice rotation and the subject each character owns.

The channel rule is that consecutive videos must not repeat: video 1 is one
character on one subject, video 2 is a different character on a different
subject, and so on. Because each voice declares its own ``subject``, that
reduces to "the next pick must be a different voice" -- which is what these
tests pin down, including the part that is easy to get wrong: the cursor has to
survive a new process, or every separate run opens on the same character.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from src.agents.voice_cast import agent as cast
from src.agents.voice_cast.voices import VOICES, get_enabled_voices

ENABLED = [vid for vid, v in VOICES.items() if v.get("enabled")]


@pytest.fixture
def rotation_file(tmp_path):
    return str(tmp_path / "voice_rotation.json")


def _pick_in_subprocess(rotation_file: str) -> str:
    """Pick a voice from a cold interpreter, as a separate CLI run would."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    code = ("from src.agents.voice_cast import agent as c;"
            f"print(c.pick_voice(path={rotation_file!r})[0])")
    out = subprocess.run([sys.executable, "-c", code], cwd=root, check=True,
                         capture_output=True, text=True)
    return out.stdout.strip()


# --- subjects ----------------------------------------------------------------

class TestSubjects:
    def test_every_enabled_voice_declares_a_subject(self):
        for vid, voice in get_enabled_voices():
            assert voice.get("subject"), f"{vid} has no subject for the joke prompt"

    def test_enabled_subjects_are_all_different(self):
        # Two characters sharing a subject would break the rotation rule at the
        # content level even though the voices differ.
        subjects = [v["subject"] for _, v in get_enabled_voices()]
        assert len(set(subjects)) == len(subjects)

    def test_the_expected_character_pairs_are_intact(self):
        assert VOICES["donald-trump"]["subject"] == "War and military strategy"
        assert VOICES["arnold-schwarzenegger"]["subject"] == "Weight lifting and bodybuilding"
        # Was "Money", which contradicted the prompt: he is about life and
        # self-improvement. The subject only labels the video.
        assert VOICES["andrew-tate"]["subject"] == "Life and self-improvement"

    def test_andrew_tate_does_not_borrow_arnolds_style(self):
        # Was "writing_style": "arnold" on the Tate entry, so he was logged and
        # described as General Arnold.
        assert VOICES["andrew-tate"]["writing_style"] == "andrew_tate"
        assert VOICES["andrew-tate"]["writing_style"] != VOICES["arnold-schwarzenegger"]["writing_style"]


# --- rotation ----------------------------------------------------------------

class TestRotation:
    def test_the_returned_entry_is_the_voice_dict_not_a_tuple(self, rotation_file):
        # Regression: automatic rotation returned the whole (id, voice) pair
        # from get_enabled_voices(), so create_video passed a tuple to
        # get_writing_style() and died with AttributeError on a live run. Every
        # other test discards the second value, so it has to be checked here.
        vid, entry = cast.pick_voice(path=rotation_file)
        assert isinstance(entry, dict), f"expected a voice dict, got {type(entry).__name__}"
        assert entry is not None
        assert entry.get("subject"), "the voice dict must carry its subject"
        assert cast.get_writing_style(vid, entry).get("persona")

    def test_both_return_shapes_agree(self, rotation_file):
        # The override branch already returned a dict; the two paths must match.
        _, forced = cast.pick_voice(preferred="donald-trump", path=rotation_file)
        _, rotated = cast.pick_voice(path=rotation_file)
        assert isinstance(forced, dict) and isinstance(rotated, dict)
        assert set(forced) == set(rotated)

    def test_consecutive_picks_are_different_characters(self, rotation_file):
        picks = [cast.pick_voice(path=rotation_file)[0] for _ in range(len(ENABLED) * 2)]
        assert all(a != b for a, b in zip(picks, picks[1:]))

    def test_rotation_cycles_through_every_enabled_voice(self, rotation_file):
        picks = [cast.pick_voice(path=rotation_file)[0] for _ in range(len(ENABLED))]
        assert sorted(picks) == sorted(ENABLED)

    def test_consecutive_picks_also_differ_in_subject(self, rotation_file):
        # The user-facing rule: video N+1 must not repeat video N's subject.
        seen = [VOICES[cast.pick_voice(path=rotation_file)[0]]["subject"]
                for _ in range(len(ENABLED) * 2)]
        assert all(a != b for a, b in zip(seen, seen[1:]))

    def test_the_cursor_survives_a_new_process(self, rotation_file):
        # A batch shares one process, but separate `python src/main.py` runs do
        # not. An in-memory counter would reset and every run would open on the
        # same character, so this genuinely spawns a cold interpreter.
        first = cast.pick_voice(path=rotation_file)[0]
        second = _pick_in_subprocess(rotation_file)
        assert first != second

    def test_a_cold_run_still_advances_the_cycle(self, rotation_file):
        # Two independent interpreters, each picking once, must also differ.
        assert _pick_in_subprocess(rotation_file) != _pick_in_subprocess(rotation_file)

    def test_state_file_is_inside_src_by_default(self):
        # Read the declared default from a cold interpreter: conftest redirects
        # ROTATION_FILE to tmp_path for every test, so the patched module
        # global cannot answer this. It must stay under src/ because
        # cleanup_temp() is allowed to wipe anything outside it.
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        code = ("from src.agents.voice_cast import agent as c;"
                "print(c.ROTATION_FILE)")
        out = subprocess.run([sys.executable, "-c", code], cwd=root, check=True,
                             capture_output=True, text=True).stdout.strip()
        assert out.startswith("src/")

    def test_the_last_voice_is_recorded_on_disk(self, rotation_file):
        vid, _ = cast.pick_voice(path=rotation_file)
        assert json.load(open(rotation_file))["last_voice"] == vid

    def test_a_preferred_voice_still_advances_the_cursor(self, rotation_file):
        target = ENABLED[-1]
        assert cast.pick_voice(preferred=target, path=rotation_file)[0] == target
        # The next automatic pick must not be the overridden voice again.
        assert cast.pick_voice(path=rotation_file)[0] != target

    def test_an_unknown_preferred_voice_falls_back_to_the_rotation(self, rotation_file, capsys):
        vid, _ = cast.pick_voice(preferred="no-such-voice", path=rotation_file)
        assert vid in ENABLED
        assert "Unknown or disabled" in capsys.readouterr().out

    def test_an_unknown_previous_voice_restarts_the_cycle(self, rotation_file):
        with open(rotation_file, "w") as f:
            json.dump({"last_voice": "a-voice-that-was-deleted"}, f)
        assert cast.pick_voice(path=rotation_file)[0] == ENABLED[0]

    def test_a_corrupt_state_file_restarts_the_cycle(self, rotation_file):
        with open(rotation_file, "w") as f:
            f.write("{not json")
        assert cast.pick_voice(path=rotation_file)[0] == ENABLED[0]

    def test_a_missing_state_file_starts_at_the_front(self, tmp_path):
        assert cast.pick_voice(path=str(tmp_path / "nope.json"))[0] == ENABLED[0]

    def test_it_raises_when_no_voice_is_enabled(self, rotation_file, monkeypatch):
        monkeypatch.setattr(cast, "get_enabled_voices", list)
        with pytest.raises(RuntimeError, match="no enabled voices"):
            cast.pick_voice(path=rotation_file)
