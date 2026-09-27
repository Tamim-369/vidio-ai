"""Publishing: what metadata reaches YouTube.

A planned video builds its own title, description and tags from the quotes it
already contains. That path is the one every batch takes, and the thing worth
testing is that it is unique per video and makes no model call.
"""
import pytest

from src.agents.publish import youtube as yt
from src.agents.publish.metadata import (
    build_description, build_tags, deterministic_metadata,
)
from src.agents.voice_cast.voices import VOICES

ARNOLD = VOICES["arnold-schwarzenegger"]
TATE = VOICES["andrew-tate"]


def _line(i, voice, text):
    return {"id": i, "text": text, "voice": voice,
            "character": voice.get("pseudonym", "")}


def _script(lines, subject="money", title=None):
    return {"lines": lines, "short_subject": subject,
            "title": title if title is not None
            else f"{lines[0]['voice']['pseudonym']}: {lines[0]['text']}"}


@pytest.fixture
def uploaded(monkeypatch):
    """Stub the network and capture what upload_video was handed."""
    calls = []
    monkeypatch.setattr(yt, "get_credentials", lambda: "creds")
    monkeypatch.setattr(yt, "build_client", lambda creds: "client")
    monkeypatch.setattr(yt, "upload_video",
                        lambda client, path, **k: calls.append(k) or "vid123")
    monkeypatch.setattr(yt, "wait_for_processing", lambda client, vid: None)
    return calls


class TestPlannedVideoSkipsTheModel:
    def test_no_metadata_call_happens(self, uploaded, monkeypatch):
        # The old path cost a generate plus up to two verifier rounds per
        # video, and could write the same description twice in a batch.
        def boom(*a, **k):
            raise AssertionError("a planned video must not call the LLM")
        monkeypatch.setattr(yt, "generate_metadata", boom)
        script = _script([_line(1, ARNOLD, "A line about the barbell.")])
        yt.publish_video("v.mp4", "money", script)
        assert uploaded[0]["title"].startswith("Brolexander: ")

    def test_a_script_without_a_title_still_falls_back(self, uploaded,
                                                      monkeypatch):
        seen = []
        monkeypatch.setattr(yt, "generate_metadata",
                            lambda t, s: seen.append(1) or {
                                "title": "LLM Title", "description": "d",
                                "tags": ["x"]})
        yt.publish_video("v.mp4", "money", {"lines": [], "topic": "money"})
        assert seen == [1]
        assert uploaded[0]["title"] == "LLM Title"


class TestDescriptionIsUniqueByConstruction:
    def test_each_line_is_quoted_under_its_own_pseudonym(self):
        script = _script([
            _line(1, ARNOLD, "First line here."),
            _line(2, TATE, "Second line here."),
        ])
        desc = build_description(script)
        assert "Brolexander: “First line here.”" in desc
        assert "Andru Tatte: “Second line here.”" in desc

    def test_it_states_the_quotes_are_parody(self):
        # A viewer who finds one of these on a channel that looks like the real
        # person's should be able to tell within a second that it is not real.
        desc = build_description(_script([_line(1, ARNOLD, "A line.")]))
        assert "Parody" in desc and "Not real" in desc

    def test_a_solo_video_names_the_speaker_and_their_topic(self):
        assert "Brolexander" in build_description(
            _script([_line(1, ARNOLD, "A line.")], subject="money"))
        assert "gains" in build_description(
            _script([_line(1, ARNOLD, "A line.")]))

    def test_the_opener_names_the_speaker_and_their_own_topic(self):
        # One character per video, so the topic claimed in the opener is always
        # the speaker's own subject and always true.
        desc = build_description(_script([_line(1, ARNOLD, "A line.")]))
        assert desc.splitlines()[0] == "Brolexander on gains."

    def test_it_hashtags_every_speaker(self):
        desc = build_description(_script([
            _line(1, ARNOLD, "A line."), _line(2, TATE, "B line."),
        ]))
        assert "#Brolexander" in desc and "#AndruTatte" in desc

    def test_two_videos_with_different_quotes_never_collide(self):
        a = build_description(_script([_line(1, ARNOLD, "Alpha quote line.")]))
        b = build_description(_script([_line(1, ARNOLD, "Beta quote line.")]))
        assert a != b

    def test_a_script_with_no_lines_still_says_something(self):
        assert build_description({"lines": []})

    def test_blank_lines_are_skipped(self):
        desc = build_description({"lines": [
            {"id": 1, "text": "   ", "voice": ARNOLD},
            {"id": 2, "text": "Real line.", "voice": ARNOLD},
        ]})
        assert "Real line." in desc and "“   ”" not in desc


class TestTags:
    def test_the_pseudonym_leads(self):
        # The pseudonyms are what a viewer of this channel searches for.
        assert build_tags(_script([_line(1, ARNOLD, "A line.")]))[0] == "brolexander"

    def test_the_real_name_is_absent(self):
        tags = " ".join(build_tags(_script([_line(1, ARNOLD, "A line.")])))
        assert "arnold" not in tags and "schwarzenegger" not in tags

    def test_each_speakers_own_topic_is_a_tag(self):
        # Each speaker contributes their own subject, so a two-person video is
        # tagged with both -- which is both what the video contains and what a
        # viewer searches. Read the expectation off the registry so a subject
        # rename cannot leave this asserting a topic nobody generates.
        tags = build_tags(_script([_line(1, ARNOLD, "A line."),
                                   _line(2, TATE, "A line.")]))
        for voice in (ARNOLD, TATE):
            assert voice["short_subject"] in tags

    def test_the_count_stays_inside_youtubes_limit(self):
        tags = build_tags(_script([
            _line(1, ARNOLD, "A."), _line(2, TATE, "B."), _line(3, VOICES["donald-trump"], "C."),
        ]))
        assert len(tags) <= 15

    def test_a_repeated_speaker_is_not_listed_twice(self):
        tags = build_tags(_script([
            _line(1, ARNOLD, "A."), _line(2, ARNOLD, "B."),
        ]))
        assert len([t for t in tags if t == "brolexander"]) == 1


class TestDeterministicMetadata:
    def test_it_returns_all_three_fields(self):
        m = deterministic_metadata(_script([_line(1, ARNOLD, "A line.")]))
        assert set(m) == {"title", "description", "tags"}

