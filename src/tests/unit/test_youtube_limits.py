"""YouTube's upload limits.

Every one of these is enforced by the API at upload time, not by anything in
this repo, so a violation is invisible until a batch is already half-published.
The title one is not hypothetical: real quotes from this channel run 58-98
characters, so a pseudonym prefix pushes a quarter of them over the limit.
"""
from src.agents.publish.metadata import _fallback_metadata, deterministic_metadata
from src.agents.video.title import TITLE_MAX_CHARS, build_title
from src.agents.voice_cast.voices import VOICES

# The API's own limits. Changing these means the API changed, not a preference.
YOUTUBE_TITLE_MAX = 100
YOUTUBE_DESCRIPTION_MAX = 5000
YOUTUBE_TAGS_MAX_CHARS = 500

PARODY = [v for v in VOICES.values() if v.get("enabled")]

# These limits are calibrated against real generated quotes, so every test here
# takes the quote_pool fixture rather than a local stub: it is the real pool
# where this machine has one, and a frozen sample of the same length range on a
# fresh clone, since used_quotes.json is gitignored run history.
def _script(quote_pool):
    lines = [{"id": i, "text": q, "voice": v}
             for i, (q, v) in enumerate(zip(quote_pool[:3], PARODY * 2), 1)]
    return {"lines": lines, "short_subject": "money",
            "title": build_title(lines[0]["text"], "andrew-tate")}


class TestTitleLimit:
    def test_the_cap_is_youtubes_limit(self):
        assert TITLE_MAX_CHARS == YOUTUBE_TITLE_MAX

    def test_every_generated_title_fits(self, quote_pool):
        # The regression that matters: a template title was always short, but a
        # quote-derived one is not, and 1 in 4 exceeded the API limit.
        for quote in quote_pool:
            for voice in PARODY:
                assert len(build_title(quote, _id_of(voice))) <= YOUTUBE_TITLE_MAX

    def test_a_wordy_quote_is_still_trimmed_to_fit(self):
        title = build_title(" ".join(["discipline"] * 200), "donald-trump")
        assert len(title) <= YOUTUBE_TITLE_MAX

    def test_trimming_never_leaves_a_partial_word(self):
        # Cutting mid-word is the one option guaranteed to look wrong, so every
        # surviving token must be a complete word from the source.
        source = "Supercalifragilistic expialidocious discipline " * 20
        title = build_title(source, "donald-trump")
        assert title.endswith("…"), "this case should actually have trimmed"
        body = title.split(": ", 1)[1].rstrip("…")   # drop the pseudonym prefix
        words = set(source.split())
        for token in body.split():
            assert token in words, f"partial word: {token!r}"

    def test_the_pseudonym_survives_trimming(self, quote_pool):
        # The name is the searchable part; it must never be what gets cut.
        for quote in quote_pool:
            for voice in PARODY:
                t = build_title(quote, _id_of(voice))
                assert t.startswith(voice["pseudonym"] + ":")

    def test_the_fallback_title_also_fits(self):
        m = _fallback_metadata("A very long topic " * 30, "1. A line.")
        assert len(m["title"]) <= YOUTUBE_TITLE_MAX

    def test_a_line_with_no_voice_id_still_gets_a_named_title(self):
        # A voice config carries the pseudonym but no "id", so resolving the
        # name by id alone would publish a bare quote over a description that
        # still says who is speaking.
        line = {"id": 1, "text": "Paying bills is stressful, so I stack them.",
                "voice": {"pseudonym": "Don Tzu"}}
        m = deterministic_metadata({"lines": [line], "short_subject": "money"})
        assert m["title"].startswith("Don Tzu: ")
        assert "Don Tzu" in m["description"]

    def test_a_title_inherited_from_an_older_script_is_still_capped(self):
        # The publisher trusts nothing upstream: a script written before the
        # limit existed arrives with a 140-character title, and YouTube rejects
        # the entire upload over the 101st character.
        stale = "Don Tzu: " + "war is a set and I bench it between sets. " * 3
        assert len(stale) > YOUTUBE_TITLE_MAX
        m = deterministic_metadata({"lines": [{"id": 1, "text": "a line",
                                               "voice": PARODY[0]}],
                                    "title": stale, "short_subject": "money"})
        assert len(m["title"]) <= YOUTUBE_TITLE_MAX
        assert m["title"].startswith("Don Tzu: ")
        assert m["title"].endswith("\u2026")


def _id_of(cfg):
    for vid, v in VOICES.items():
        if v is cfg:
            return vid
    raise AssertionError("voice not in registry")


class TestDescriptionLimit:
    def test_a_full_description_fits(self, quote_pool):
        script = {"lines": [{"id": i, "text": q, "voice": v}
                            for i, (q, v) in enumerate(zip(quote_pool[:3], PARODY * 2), 1)],
                  "short_subject": "money", "title": "x"}
        d = deterministic_metadata(script)["description"]
        assert len(d) <= YOUTUBE_DESCRIPTION_MAX

    def test_the_fallback_description_fits(self):
        m = _fallback_metadata("topic", "\n".join(["a line"] * 500))
        assert len(m["description"]) <= YOUTUBE_DESCRIPTION_MAX


class TestTagsLimit:
    def test_the_tag_payload_fits(self, quote_pool):
        script = _script(quote_pool)
        tags = deterministic_metadata(script)["tags"]
        assert len(",".join(tags)) <= YOUTUBE_TAGS_MAX_CHARS

    def test_a_three_character_video_still_fits(self, quote_pool):
        lines = [{"id": i, "text": q, "voice": v}
                 for i, (q, v) in enumerate(zip(quote_pool[:3], PARODY), 1)]
        tags = deterministic_metadata({"lines": lines, "short_subject": "money",
                                       "title": "x"})["tags"]
        assert len(",".join(tags)) <= YOUTUBE_TAGS_MAX_CHARS

    def test_the_fallback_tags_fit(self):
        m = _fallback_metadata("x", "1. a")
        assert len(",".join(m["tags"])) <= YOUTUBE_TAGS_MAX_CHARS
