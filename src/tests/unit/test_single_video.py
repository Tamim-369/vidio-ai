"""One character, one quote, one video: the fan-out and the card.

The pipeline hands the renderer a single line, so the interesting questions are
narrow: that the line survives the worker split exactly once and in order, that the
face is validated before anything renders rather than halfway through, and that
a line whose TTS produced nothing is an error rather than a blank video. Every
one of those fails silently -- you get a short or empty video, not a traceback.
"""
import pytest

from src.agents.visuals import card
from src.agents.voiceover.workers import _split_chunks, _split_chunks_by_voice

DON = {"name": "Donald Trump", "ref_audio": "trump.wav", "face": "Trump.png",
       "engine": "chatterbox"}


def _line(i, voice=None, text=None):
    return {"id": i, "text": text or f"line {i}", "character": voice["name"],
            "voice": voice}


class TestWorkerSplit:
    def test_one_line_stays_a_single_chunk(self):
        # The live shape: one quote, so however many workers are offered, the
        # line must not be duplicated or dropped.
        line = _line(1, DON)
        assert _split_chunks_by_voice([line], 4) == [[line]]

    def test_every_line_is_rendered_exactly_once(self):
        lines = [_line(i, DON) for i in range(1, 6)]
        chunks = _split_chunks_by_voice(lines, 2)
        assert sorted(l["id"] for c in chunks for l in c) == [1, 2, 3, 4, 5]

    def test_a_monologue_falls_back_to_an_even_split(self):
        # One character throughout is the normal case, so this must not end up
        # as one chunk of 3 beside an empty one.
        lines = [_line(i, DON) for i in range(1, 4)]
        assert [len(c) for c in _split_chunks_by_voice(lines, 2)] == [2, 1]

    def test_the_plain_split_is_still_ordered(self):
        # Reordering is invisible in the output file, so it is asserted here.
        lines = [{"id": i} for i in range(1, 6)]
        chunks = _split_chunks(lines, 2)
        assert [l["id"] for l in chunks[0]] == [1, 2, 3]
        assert [l["id"] for l in chunks[1]] == [4, 5]

    def test_lines_without_their_own_voice_use_the_default(self):
        chunks = _split_chunks_by_voice([{"id": 1, "text": "a"}, {"id": 2, "text": "b"}],
                                        2, default_voice=DON)
        assert len(chunks) == 2

    def test_no_lines_makes_no_chunks(self):
        assert _split_chunks_by_voice([], 2) == []


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """Stub the card renderer and the concat, recording what each was handed."""
    used = []
    shots = []

    def fake_render_card(quote, image_path, audio_path, out_path, *a, **k):
        used.append((quote, image_path))
        with open(out_path, "wb") as fh:
            fh.write(b"card")
        return out_path

    def fake_concat(paths, out_path, *a, **k):
        shots.append(list(paths))
        with open(out_path, "wb") as fh:
            fh.write(b"video")
        return out_path

    monkeypatch.setattr(card, "render_card", fake_render_card)
    monkeypatch.setattr(card, "concat_segments", fake_concat)
    monkeypatch.setattr(card, "output_path", lambda topic: str(tmp_path / "cards.mp4"))
    monkeypatch.setattr(card, "TEMP_DIR", str(tmp_path))
    return shots, used


def _voice_and_face(tmp_path):
    face = tmp_path / "trump.png"
    face.write_bytes(b"png")
    voice = dict(DON, face=str(face), authors=["Donald Trump"])
    return voice, face


def _rendered(line, voice, face, tmp_path, audio=True):
    return {"id": 1, "text": line["text"], "character": voice["name"], "voice": voice,
            "audio_path": str(tmp_path / "1.wav") if audio else None,
            "actual_duration": 4.0}


def test_a_single_character_video_renders_one_card(stub, tmp_path):
    _, used = stub
    voice, face = _voice_and_face(tmp_path)
    card.render({"lines": [_rendered(_line(1, voice), voice, face, tmp_path)],
                 "topic": "war", "title": "t"})
    assert len(used) == 1, "expected exactly one card for one quote"
    assert used[0][1] == str(face), "the card drew the wrong portrait"


def test_a_missing_face_is_caught_before_anything_renders(stub, tmp_path):
    # Validating halfway through leaves a half-rendered temp dir and a confusing
    # error, so the check has to come first.
    _, used = stub
    voice, _ = _voice_and_face(tmp_path)
    voice["face"] = str(tmp_path / "absent.png")
    with pytest.raises(RuntimeError, match="face"):
        card.render({"lines": [_rendered(_line(1, voice), voice, None, tmp_path)],
                     "topic": "war", "title": "t"})
    assert used == [], "validation must happen before the first card"

def test_a_script_with_nothing_to_render_is_an_error(stub, tmp_path):
    _, used = stub
    voice, face = _voice_and_face(tmp_path)
    with pytest.raises(RuntimeError, match="[Nn]o renderable"):
        card.render({"lines": [_rendered(_line(1, voice), voice, face, tmp_path,
                                         audio=False)],
                     "topic": "war", "title": "t"})
    assert used == []
