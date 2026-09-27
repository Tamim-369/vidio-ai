"""Multi-character narration: who speaks which line, and with whose face.

A 2- or 3-character video puts a different voice on every line, so these cover
the three places that has to hold: the fan-out splits lines by character, the
engine refuses to mix TTS backends, and the card draws the right face per line.
"""
import os

import pytest

from src.agents.visuals import card
from src.agents.voiceover import engine
from src.agents.voiceover.workers import _split_chunks, _split_chunks_by_voice

DON = {"name": "Donald Trump", "ref_audio": "trump.wav", "face": "Trump.png",
       "engine": "chatterbox"}
TATE = {"name": "Andrew Tate", "ref_audio": "tate.wav", "face": "Tate.png",
        "engine": "chatterbox"}
ARNOLD = {"name": "Arnold Schwarzenegger", "ref_audio": "arnold.wav",
          "face": "Arnold.png", "engine": "chatterbox"}


def _line(i, voice, text=None):
    return {"id": i, "text": text or f"line {i}", "character": voice["name"],
            "voice": voice}


def _chunks_of(chunks, name):
    return {n for c in chunks for n in [x["character"] for x in c
                                       if x["voice"] is name]}


class TestSplitChunksByVoice:
    def test_a_clash_gives_each_worker_one_character(self):
        lines = [_line(1, DON), _line(2, TATE)]
        chunks = _split_chunks_by_voice(lines, 2)
        # The whole point: no worker holds both characters, so neither waits on
        # the other's voice conditionals.
        assert len(chunks) == 2
        assert all(len({l["character"] for l in c}) == 1 for c in chunks)

    def test_every_line_is_rendered_exactly_once(self):
        lines = [_line(i, DON) for i in range(1, 5)] + [_line(5, TATE)]
        chunks = _split_chunks_by_voice(lines, 2)
        assert sorted(l["id"] for c in chunks for l in c) == [1, 2, 3, 4, 5]

    def test_a_monologue_falls_back_to_an_even_split(self):
        # Three lines, one character: there is nothing to keep apart, so this
        # must not end up as one chunk of 3 and one empty chunk.
        lines = [_line(i, DON) for i in range(1, 4)]
        chunks = _split_chunks_by_voice(lines, 2)
        assert [len(c) for c in chunks] == [2, 1]

    def test_three_characters_over_two_workers_still_covers_everything(self):
        lines = [_line(1, DON), _line(2, TATE), _line(3, ARNOLD)]
        chunks = _split_chunks_by_voice(lines, 2)
        assert sorted(l["id"] for c in chunks for l in c) == [1, 2, 3]

    def test_a_lopsided_cast_stays_balanced(self):
        # One character with 3 lines and another with 1 over 2 workers: the
        # bigger group takes a bin to itself so the workers finish together.
        lines = [_line(1, DON), _line(2, DON), _line(3, DON), _line(4, TATE)]
        chunks = _split_chunks_by_voice(lines, 2)
        assert sorted(len(c) for c in chunks) == [1, 3]

    def test_lines_without_their_own_voice_use_the_default(self):
        lines = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
        chunks = _split_chunks_by_voice(lines, 2, default_voice=DON)
        assert len(chunks) == 2

    def test_one_line_stays_a_single_chunk(self):
        assert _split_chunks_by_voice([_line(1, DON)], 1) == [[_line(1, DON)]]

    def test_no_lines_makes_no_chunks(self):
        assert _split_chunks_by_voice([], 2) == []

    def test_the_plain_split_is_still_ordered(self):
        lines = [{"id": i} for i in range(1, 6)]
        chunks = _split_chunks(lines, 2)
        assert [l["id"] for l in chunks[0]] == [1, 2, 3]
        assert [l["id"] for l in chunks[1]] == [4, 5]


class TestEngineSelection:
    def test_refuses_to_mix_tts_engines_in_one_video(self):
        # Each backend loads its own model, so a video cannot swap mid-way.
        lines = [_line(1, {"name": "a", "engine": "chatterbox"}),
                 _line(2, {"name": "b", "engine": "pocket"})]
        with pytest.raises(RuntimeError, match="mix TTS engines"):
            engine.generate_audio(lines)

    def test_all_chatterbox_takes_the_chatterbox_path(self, monkeypatch):
        seen = []
        monkeypatch.setattr(engine, "_generate_chatterbox",
                            lambda lines, d, v=None: seen.append("chat"))
        engine.generate_audio([_line(1, DON), _line(2, TATE)])
        assert seen == ["chat"]

    def test_all_pocket_takes_the_pocket_path(self, monkeypatch):
        seen = []
        monkeypatch.setattr(engine, "_generate_pocket",
                            lambda lines, d: seen.append("pocket"))
        engine.generate_audio([{"id": 1, "text": "a"}])
        assert seen == ["pocket"]

    def test_a_voice_default_applies_to_lines_that_carry_none(self, monkeypatch):
        seen = []
        monkeypatch.setattr(engine, "_generate_chatterbox",
                            lambda lines, d, v=None: seen.append((lines, v)))
        # No engine on the default entry means pocket for the unlabelled lines.
        engine.generate_audio([{"id": 1, "text": "a"}], voice=DON)
        assert seen and seen[0][1] is DON


def _faces(tmp_path, voices):
    """Real face files on disk, since render() validates them up front."""
    for v in voices:
        f = tmp_path / v["face"]
        f.write_bytes(b"png")
        v = v
    return {v["name"]: str(tmp_path / v["face"]) for v in voices}


def _rendered(line, face, voice, tmp_path):
    return {"id": line["id"], "text": line["text"], "character": voice["name"],
            "voice": voice, "audio_path": str(tmp_path / f"{line['id']}.wav"),
            "actual_duration": 4.0}


class TestCardFacePerLine:
    @pytest.fixture
    def stub(self, monkeypatch, tmp_path):
        shots = []
        used = []

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
        monkeypatch.setattr(card, "output_path",
                            lambda topic: str(tmp_path / "cards.mp4"))
        monkeypatch.setattr(card, "TEMP_DIR", str(tmp_path))
        return shots, used

    def _voices(self, tmp_path):
        return [
            dict(DON, face=str(tmp_path / "trump.png"),
                 authors=["Donald Trump"]),
            dict(TATE, face=str(tmp_path / "tate.png"),
                 authors=["Andrew Tate"]),
            dict(ARNOLD, face=str(tmp_path / "arnold.png"),
                 authors=["Arnold Schwarzenegger"]),
        ]

    def test_each_line_is_drawn_with_its_own_face(self, monkeypatch, tmp_path,
                                                  stub):
        shots, used = stub
        don, tate, arnold = self._voices(tmp_path)
        for v in (don, tate, arnold):
            (tmp_path / v["face"]).write_bytes(b"png")
        lines = [_rendered(_line(1, don), don["face"], don, tmp_path),
                 _rendered(_line(2, tate), tate["face"], tate, tmp_path),
                 _rendered(_line(3, arnold), arnold["face"], arnold, tmp_path)]
        card.render({"lines": lines, "topic": "war", "title": "t"})

        faces = [f for _, f in used]
        assert faces == [don["face"], tate["face"], arnold["face"]]
        # A conversation drawn with one face is the giveaway it is not one.
        assert len(set(faces)) == 3

    def test_the_output_holds_one_clip_per_line(self, stub, tmp_path):
        shots, _ = stub
        don, tate, _ = self._voices(tmp_path)
        for v in (don, tate):
            (tmp_path / v["face"]).write_bytes(b"png")
        lines = [_rendered(_line(1, don), don["face"], don, tmp_path),
                 _rendered(_line(2, tate), tate["face"], tate, tmp_path)]
        card.render({"lines": lines, "topic": "war", "title": "t"})
        assert len(shots[0]) == 2

    def test_lines_keep_their_script_order(self, stub, tmp_path):
        shots, used = stub
        don, tate, _ = self._voices(tmp_path)
        for v in (don, tate):
            (tmp_path / v["face"]).write_bytes(b"png")
        lines = [_rendered(_line(1, don, "first"), don["face"], don, tmp_path),
                 _rendered(_line(2, tate, "second"), tate["face"], tate,
                           tmp_path)]
        card.render({"lines": lines, "topic": "war", "title": "t"})
        assert [q for q, _ in used] == ["first", "second"]

    def test_a_single_character_video_still_renders(self, stub, tmp_path):
        shots, used = stub
        don, _, _ = self._voices(tmp_path)
        (tmp_path / don["face"]).write_bytes(b"png")
        card.render({"lines": [_rendered(_line(1, don), don["face"], don,
                                         tmp_path)],
                     "topic": "war", "title": "t"})
        assert len(used) == 1

    def test_a_missing_face_is_caught_before_anything_renders(self, stub,
                                                              tmp_path):
        shots, used = stub
        don, tate, _ = self._voices(tmp_path)
        (tmp_path / don["face"]).write_bytes(b"png")  # tate's face is absent
        lines = [_rendered(_line(1, don), don["face"], don, tmp_path),
                 _rendered(_line(2, tate), tate["face"], tate, tmp_path)]
        with pytest.raises(RuntimeError, match="face"):
            card.render({"lines": lines, "topic": "war", "title": "t"})
        assert used == [], "validation must happen before the first card"

    def test_a_line_without_audio_is_skipped_not_fatal(self, stub, tmp_path):
        # TTS can fail one line of three; the other two should still publish.
        shots, used = stub
        don, tate, _ = self._voices(tmp_path)
        for v in (don, tate):
            (tmp_path / v["face"]).write_bytes(b"png")
        silent = _rendered(_line(1, don), don["face"], don, tmp_path)
        silent["audio_path"] = None
        ok = _rendered(_line(2, tate), tate["face"], tate, tmp_path)
        card.render({"lines": [silent, ok], "topic": "war", "title": "t"})
        assert [q for q, _ in used] == [ok["text"]]

    def test_a_script_with_nothing_to_render_is_an_error(self, stub, tmp_path):
        shots, used = stub
        don, _, _ = self._voices(tmp_path)
        (tmp_path / don["face"]).write_bytes(b"png")
        silent = _rendered(_line(1, don), don["face"], don, tmp_path)
        silent["audio_path"] = None
        with pytest.raises(RuntimeError, match="[Nn]o renderable"):
            card.render({"lines": [silent], "topic": "war", "title": "t"})
