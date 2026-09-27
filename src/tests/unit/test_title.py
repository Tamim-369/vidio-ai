"""Titles: the channel's pseudonym plus the quote, inside YouTube's limit."""
from src.agents.voice_cast.voices import get_enabled_voices
from src.agents.video.title import TITLE_MAX_CHARS, build_title

ALL = [vid for vid, _ in get_enabled_voices()]


class TestTitles:
    """Titles are the channel's pseudonym plus that video's own first line."""

    Q = "Paying bills is stressful. That is why I stack them like plates."

    def test_the_title_leads_with_the_pseudonym(self):
        assert build_title(self.Q, "donald-trump").startswith("Don Tzu: ")

    def test_the_real_name_never_appears(self):
        # These are fabricated quotes in the voices of living people, so the
        # parody name is the one that belongs in metadata.
        for vid, real in (("donald-trump", "Donald"), ("andrew-tate", "Andrew"),
                          ("arnold-schwarzenegger", "Arnold")):
            assert real not in build_title(self.Q, vid)

    def test_the_quote_is_carried_in_full(self):
        # Cutting to the feed's 40-char window would slice off the punchline,
        # which is the second sentence of every quote this channel writes.
        assert self.Q in build_title(self.Q, "donald-trump")

    def test_different_quotes_give_different_titles(self):
        a = build_title(self.Q, "donald-trump")
        b = build_title("Discipline is a set. I bench it between sets.", "donald-trump")
        assert a != b

    def test_each_voice_uses_its_own_pseudonym(self):
        for vid, name in (("donald-trump", "Don Tzu"),
                          ("andrew-tate", "Andru Tatte"),
                          ("arnold-schwarzenegger", "Brolexander")):
            assert build_title(self.Q, vid).startswith(f"{name}: ")

    def test_an_unknown_voice_still_produces_a_title(self):
        assert build_title(self.Q, "nobody") and self.Q in build_title(self.Q, "nobody")

    def test_pathological_length_is_still_capped(self):
        long_quote = "word " * 200
        assert len(build_title(long_quote, "donald-trump")) <= TITLE_MAX_CHARS

    def test_is_deterministic(self):
        for vid in ALL:
            assert build_title(self.Q, vid) == build_title(self.Q, vid)

    def test_a_quote_that_fits_keeps_the_whole_joke(self):
        # Quotes run 58-98 characters and the pseudonym costs ~13, so most
        # titles land under YouTube's 100-character limit intact. Only the
        # longest lose their tail, and only down to that limit.
        short = "Paying bills is stressful. That's why I stack them like plates."
        assert build_title(short, "andrew-tate").endswith(short)

    def test_a_quote_that_overflows_loses_only_its_tail(self):
        long_q = ("Discipline is the only shortcut. " * 6).strip()
        title = build_title(long_q, "andrew-tate")
        assert len(title) <= TITLE_MAX_CHARS
        assert title.startswith("Andru Tatte: ")
        assert long_q.startswith(title.split(": ", 1)[1].rstrip("…"))


class TestTitleUniqueness:
    def test_the_quote_pool_makes_titles_unique_without_a_template(self, quote_pool):
        # The property the template version could not give: a format-derived
        # title repeats, a quote-derived one cannot, because the pool is
        # deduplicated before a video is ever planned. The fixture is the real
        # pool where it exists and a frozen sample of the same shape otherwise,
        # because used_quotes.json is gitignored and absent on a fresh clone.
        assert len(quote_pool) >= 2, "need real quotes for this to mean anything"
        seen = set()
        for text in quote_pool[:80]:
            t = build_title(text, "donald-trump")
            assert t not in seen
            seen.add(t)

    def test_three_videos_with_different_quotes_never_collide(self):
        qs = [f"A distinct quote number {i} with enough length to be real." for i in range(3)]
        titles = [build_title(q, "andrew-tate") for q in qs]
        assert len(set(titles)) == 3
