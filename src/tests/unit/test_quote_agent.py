"""Tests for the quote agent.

The network path is stubbed: every test here exercises parsing, filtering,
rotation and persistence against a faked llm.call_groq, so the suite costs no
Groq calls and runs offline.
"""
from __future__ import annotations

import json

import pytest

from src.services import quote_agent
from src.services.quote_agent import Quote, _Pool, generate_quotes


@pytest.fixture
def pool_file(tmp_path):
    return str(tmp_path / "used_quotes.json")


def _reply(*quotes):
    """Build a model reply in the requested JSON array format."""
    return json.dumps([{"quote": q, "format": "one_liner", "source": ""} for q in quotes])


# --- parsing -----------------------------------------------------------------

class TestParseCandidates:
    def test_parses_requested_json_array(self):
        got = quote_agent.parse_candidates(_reply("The very first mock quote in this batch.", "The second mock quote in this batch here."))
        assert [c["quote"] for c in got] == ["The very first mock quote in this batch.", "The second mock quote in this batch here."]

    def test_reads_bare_array(self):
        assert quote_agent.parse_candidates('["A bare string mock quote, not an object."]')[0]["quote"] == \
            "A bare string mock quote, not an object."

    def test_reads_fenced_json(self):
        raw = '```json\n[{"quote": "A fenced mock quote right here."}]\n```'
        assert quote_agent.parse_candidates(raw)[0]["quote"] == "A fenced mock quote right here."

    def test_reads_single_object(self):
        raw = '{"quote": "A single object mock quote for the parser."}'
        assert quote_agent.parse_candidates(raw)[0]["quote"] == "A single object mock quote for the parser."

    def test_plain_text_fallback_splits_lines(self):
        raw = "A plain first quote.\nA plain second quote."
        got = quote_agent.parse_candidates(raw)
        assert len(got) == 2

    def test_strips_numbering_and_quotes(self):
        assert quote_agent.parse_candidates('1. "A numbered mock quote in this batch here."')[0]["quote"] == \
            "A numbered mock quote in this batch here."

    def test_strips_bullets(self):
        got = quote_agent.parse_candidates('* A bulleted mock quote right here.')
        assert got[0]["quote"] == "A bulleted mock quote right here."

    def test_empty_reply_yields_nothing(self):
        assert quote_agent.parse_candidates("") == []
        assert quote_agent.parse_candidates("   ") == []
        assert quote_agent.parse_candidates(None) == []

    def test_format_and_source_are_lowercased(self):
        raw = '[{"quote": "A short mock quote for the parser tests.", "format": "One_Liner", "source": "Sun Tzu"}]'
        got = quote_agent.parse_candidates(raw)
        assert got[0]["format"] == "one_liner"
        assert got[0]["source"] == "Sun Tzu"

    def test_plain_text_drops_meta_lines(self):
        raw = "Real quote number one here.\nSelf-check: this one is fine."
        got = quote_agent.parse_candidates(raw)
        assert len(got) == 1


# --- rejection ---------------------------------------------------------------

class TestRejectReason:
    def test_accepts_a_plain_quote(self):
        cand = {"quote": "The tank politely exploded, again, on schedule.", "format": "one_liner"}
        assert quote_agent.reject_reason(cand, _Pool()) == ""

    @pytest.mark.parametrize("text,reason", [
        ("Self-check: does it make sense?", "meta-commentary"),
        ("As an AI I must note this is fake", "meta-commentary"),
        ("note: regenerated a better one", "meta-commentary"),
        ("Too short.", "too short"),
        ("x" * 400, "too long"),
        ("first beat here\nsecond beat here", "multi-line (must be one beat)"),
    ])
    def test_rejects_bad_candidates(self, text, reason):
        assert quote_agent.reject_reason({"quote": text}, _Pool()) == reason

    def test_rejects_empty(self):
        assert quote_agent.reject_reason({"quote": "  "}, _Pool()) == "empty"

    # --- 30-80 character card budget ---

    def test_the_length_window_is_30_to_80(self):
        assert quote_agent.MIN_QUOTE_CHARS == 30
        assert quote_agent.MAX_QUOTE_CHARS == 80

    def test_an_80_char_quote_is_accepted(self):
        text = "x" * 80
        assert quote_agent.reject_reason(
            {"quote": text, "format": "one_liner"}, _Pool()) == ""

    def test_81_chars_is_rejected(self):
        text = "x" * 81
        assert quote_agent.reject_reason({"quote": text}, _Pool()) == "too long"

    def test_a_30_char_quote_is_accepted(self):
        text = "x" * 30
        assert quote_agent.reject_reason(
            {"quote": text, "format": "one_liner"}, _Pool()) == ""

    def test_29_chars_is_rejected(self):
        # The floor exists so a terse line does not read as a bare caption.
        text = "x" * 29
        assert quote_agent.reject_reason({"quote": text}, _Pool()) == "too short"

    def test_the_bounds_count_spaces_and_punctuation(self):
        # Length is measured on the final spoken text, so a quote that is only
        # a character or two outside the window is still rejected.
        text = "Know thyself, or at least know where you left the laundry basket."
        assert len(text) == 65
        assert quote_agent.reject_reason({"quote": text}, _Pool()) == ""
        assert quote_agent.reject_reason({"quote": text + "x" * 16}, _Pool()) == "too long"

    def test_the_prompt_states_both_bounds(self):
        # Without the numbers in the prompt the model keeps writing 150-char
        # jokes (or 12-char one-liners) that get thrown away, starving the batch.
        from src.services.quote_agent import _build_messages

        content = _build_messages([], _Pool(), 2)[0]["content"]
        assert "between 30 and 80 characters" in content

    def test_rejects_duplicate_regardless_of_case_and_punctuation(self):
        pool = _Pool(quotes=[{"text": "The tank politely exploded, again, on schedule."}])
        cand = {"quote": "the tank politely exploded, again, on schedule"}
        assert quote_agent.reject_reason(cand, pool) == "already used"

    def test_rejects_duplicate_ignoring_extra_spacing(self):
        pool = _Pool(quotes=[{"text": "The tank politely exploded, again, on schedule."}])
        assert quote_agent.reject_reason({"quote": "The  tank   politely exploded, again, on schedule."}, pool) == \
            "already used"

    def test_rejects_unknown_format(self):
        cand = {"quote": "A perfectly fine quote here, give or take.", "format": "haiku"}
        assert quote_agent.reject_reason(cand, _Pool()) == "unknown format 'haiku'"

    def test_allows_missing_format(self):
        assert quote_agent.reject_reason({"quote": "A perfectly fine quote for the record."}, _Pool()) == ""


class TestFactRejection:
    """Real wisdom slipping through the prompt's self-check.

    Found by a live run: the model returned "The Battle of Thermopylae was a
    tactical victory, not a strategic one." — a true historical fact, exactly
    what the prompt forbids. Mechanical filters cannot judge humour, but this
    class of failure is recognisable, so it is rejected rather than narrated.
    """

    @pytest.mark.parametrize("text", [
        "The Battle of Thermopylae was a tactical victory, not a strategic one.",
        "The war was a disaster for everyone involved, not a triumph.",
        "The empire is remembered as a golden age of trade and learning.",
        "The siege was a masterpiece of logistics, not an accident.",
        "He was a general in 480 BC, not a philosopher.",
        "He died in 44 BC, apparently, which is a fact.",
    ])
    def test_rejects_historical_facts(self, text):
        assert quote_agent.reject_reason({"quote": text}, _Pool()) == \
            "reads like a fact, not a joke"

    @pytest.mark.parametrize("text", [
        # A joke that merely mentions a war or a number must survive.
        "Sun Tzu walked into a tavern and ordered 480 shots.",
        "I declare war on the last slice of pizza.",
        "The Battle of the Breakfast Table ended in syrup.",
    ])
    def test_keeps_jokes_that_mention_history(self, text):
        reason = quote_agent.reject_reason({"quote": text}, _Pool())
        assert reason != "reads like a fact, not a joke", reason

    def test_the_live_failure_is_rejected(self):
        # The exact string a real run produced.
        bad = "The Battle of Thermopylae was a tactical victory, not a strategic one."
        assert quote_agent.reject_reason({"quote": bad}, _Pool()) == \
            "reads like a fact, not a joke"


# --- figure rotation ---------------------------------------------------------

class TestFigureDetection:
    @pytest.mark.parametrize("text,source,expected", [
        ("Sun Tzu would have loved this plan.", "", "sun tzu"),
        ("A clever twist on the old line", "Socrates", "socrates"),
        ("Confucius walks into a bar.", "", "confucius"),
        ("Nobody famous said this one.", "", ""),
    ])
    def test_detects_figure(self, text, source, expected):
        assert quote_agent._detect_figure(text, source) == expected

    def test_unused_figures_are_never_stale(self):
        pool = _Pool(figures=["sun tzu"])
        assert quote_agent._figure_is_stale("socrates", pool) is False

    def test_recent_figure_is_stale_once_pool_is_exhausted(self):
        # Every figure used, and the last one was this -> prefer another.
        pool = _Pool(figures=list(quote_agent.SOURCE_FIGURES))
        assert quote_agent._figure_is_stale(pool.figures[-1], pool) is True
        assert quote_agent._figure_is_stale(pool.figures[0], pool) is False

    def test_no_figure_is_never_stale(self):
        pool = _Pool(figures=list(quote_agent.SOURCE_FIGURES))
        assert quote_agent._figure_is_stale("", pool) is False


# --- pool persistence --------------------------------------------------------

class TestPool:
    def test_missing_file_gives_empty_pool(self, pool_file):
        pool = quote_agent._load_pool(pool_file)
        assert pool.quotes == [] and pool.figures == []

    def test_corrupt_file_gives_empty_pool(self, pool_file):
        with open(pool_file, "w") as f:
            f.write("{not json")
        assert quote_agent._load_pool(pool_file).quotes == []

    def test_round_trip(self, pool_file):
        pool = _Pool()
        quote_agent._remember(pool, Quote(text="A remembered mock quote from the old pool.",
                                          figure="sun tzu"))
        quote_agent._save_pool(pool, pool_file)
        back = quote_agent._load_pool(pool_file)
        assert back.quotes[0]["text"] == "A remembered mock quote from the old pool."
        assert back.figures == ["sun tzu"]

    def test_figure_recorded_once(self, pool_file):
        pool = _Pool()
        for _ in range(3):
            quote_agent._remember(pool, Quote(text="Some perfectly usable mock quote here.", figure="plato"))
        assert pool.figures == ["plato"]

    def test_creates_parent_directory(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "pool.json")
        quote_agent._save_pool(_Pool(), path)
        assert (tmp_path / "deep" / "nested" / "pool.json").is_file()

    def test_state_file_is_inside_src_by_default(self):
        # Keeps the pool durable and out of the repo root.
        assert quote_agent.STATE_FILE.startswith("src/")


class TestPromptIntegrity:
    """The prompt is the user's specification, reproduced verbatim.

    These assertions fail if the text is reflowed, "improved", or truncated, so
    a well-meaning edit to the wording surfaces as a test failure.
    """

    VERBATIM_LINES = [
        "No real wisdom. Punchline can't be a valid point or clever logic, even ironic. "
        "If it makes actual sense, it's wrong.",
        'One punchline, one beat. No "and then," no stacked scenarios, no mini-stories.',
        "Twist must stay on the same topic as the setup — don't swap to something unrelated.",
        "Plain, spoken words only. No essay vocabulary. No meme-crutch phrases.",
        "Default short — one sentence + one punch. Only go longer for one sharp concrete "
        "detail, never a scene.",
        "When twisting a real quote/proverb, you MUST change the wording of the "
        "ending/key phrase into something dumb. Never just pair it with a second real "
        "proverb — that's still real wisdom, just doubled.",
        "When using a fake attribution, the quote itself must also be altered/made-up — "
        "never attach a fake name to an untouched real quote.",
        "No recycled meme lines — don't reuse existing internet jokes/t-shirt slogans. "
        "Generate something new.",
        "Rotate formats across a batch: twisted proverb (from the source pool above), "
        "fake attribution + altered quote, original one-liner, crude/innuendo.",
        "Output only the final quotes — no format labels, no self-correction, no "
        "meta-commentary. If a quote turns out to be real/unaltered, silently discard "
        "and regenerate instead of narrating it.",
    ]

    SELF_CHECK_LINES = [
        "Does it secretly make sense / is it actually a fair point? → fix.",
        "Is it just two real sayings paired together? → fix.",
        "Is the fake-attributed quote actually altered, not just relabeled? → fix.",
        "Is this a recycled meme I've seen before? → discard, make a new one.",
        "Any leftover labels, notes, or self-talk in the output? → strip it out.",
        "Did I pull from Sun Tzu / Greek philosophers / similar classic sources for the "
        "twisted-proverb format? → check rotation.",
    ]

    @pytest.mark.parametrize("line", VERBATIM_LINES)
    def test_rule_line_present_verbatim(self, line):
        assert line in quote_agent.QUOTE_PROMPT

    @pytest.mark.parametrize("line", SELF_CHECK_LINES)
    def test_self_check_line_present_verbatim(self, line):
        assert line in quote_agent.QUOTE_PROMPT

    def test_source_pool_is_named_in_prompt(self):
        for fig in ("Sun Tzu", "Socrates", "Confucius"):
            assert fig in quote_agent.QUOTE_PROMPT

    def test_prompt_is_not_truncated(self):
        assert len(quote_agent.QUOTE_PROMPT) > 1500

    def test_every_declared_figure_is_detectable(self):
        # SOURCE_FIGURES drives rotation, so each must actually be findable.
        for fig in quote_agent.SOURCE_FIGURES:
            assert quote_agent._detect_figure(f"{fig} said so, and nobody argued back") == fig

    def test_output_format_asks_for_a_json_array(self):
        fmt = quote_agent.OUTPUT_FORMAT.format(n=3)
        assert "JSON array" in fmt
        assert "3" in fmt


# --- generation (stubbed network) -------------------------------------------

class TestGenerateQuotes:
    def test_returns_requested_count(self, monkeypatch, pool_file):
        monkeypatch.setattr(
            quote_agent.llm, "call_groq",
            lambda *a, **k: _reply("The first valid mock quote in this batch.", "The second valid mock quote in this batch."))
        got = generate_quotes(n=2, pool_path=pool_file)
        assert len(got) == 2
        assert all(isinstance(q, Quote) for q in got)

    def test_is_groq_only(self, monkeypatch, pool_file):
        seen = {}

        def fake(messages, **k):
            seen.update(k)
            return _reply("A perfectly valid mock quote right here.")

        monkeypatch.setattr(quote_agent.llm, "call_groq", fake)
        generate_quotes(n=1, pool_path=pool_file)
        assert seen["allow_fallback"] is False
        assert seen["model"] == quote_agent.GROQ_QUOTE_MODEL

    def test_persists_accepted_quotes(self, monkeypatch, pool_file):
        monkeypatch.setattr(quote_agent.llm, "call_groq",
                            lambda *a, **k: _reply("A persisted mock quote right here, saved."))
        generate_quotes(n=1, pool_path=pool_file)
        saved = json.load(open(pool_file))
        assert saved["quotes"][0]["text"] == "A persisted mock quote right here, saved."

    def test_accumulates_across_calls_and_never_repeats(self, monkeypatch, pool_file):
        seq = iter(["Unique mock quote number one in this batch.", "Unique mock quote number two in this batch."])
        monkeypatch.setattr(quote_agent.llm, "call_groq", lambda *a, **k: _reply(next(seq)))
        first = generate_quotes(n=1, pool_path=pool_file)
        second = generate_quotes(n=1, pool_path=pool_file)
        assert first[0].text != second[0].text
        assert len(json.load(open(pool_file))["quotes"]) == 2

    def test_retries_when_everything_is_rejected(self, monkeypatch, pool_file):
        calls = iter([_reply("Self-check: fine, the joke stands as written."), _reply("A recovered valid mock quote found here.")])
        monkeypatch.setattr(quote_agent.llm, "call_groq", lambda *a, **k: next(calls))
        got = generate_quotes(n=1, pool_path=pool_file)
        assert got[0].text == "A recovered valid mock quote found here."

    def test_raises_when_no_round_produces_enough(self, monkeypatch, pool_file):
        monkeypatch.setattr(quote_agent.llm, "call_groq",
                            lambda *a, **k: _reply("Self-check: nope, but the joke is still fine."))
        with pytest.raises(RuntimeError, match="usable quotes"):
            generate_quotes(n=1, pool_path=pool_file)

    def test_rejected_text_is_fed_back(self, monkeypatch, pool_file):
        replies = iter([_reply("Self-check: rejected one, the rest is fine."),
                        _reply("A good mock quote after the feedback round.")])
        captured = []

        def fake(messages, **k):
            captured.append(messages[0]["content"])
            return next(replies)

        monkeypatch.setattr(quote_agent.llm, "call_groq", fake)
        generate_quotes(n=1, pool_path=pool_file)
        assert len(captured) == 2
        assert "rejected" in captured[1].lower()

    def test_sends_history_so_later_rounds_avoid_repeats(self, monkeypatch, pool_file):
        captured = []
        replies = iter([_reply("The first entirely unique mock quote here."), _reply("The second entirely unique mock quote.")])

        def fake(messages, **k):
            captured.append(messages[0]["content"])
            return next(replies)

        monkeypatch.setattr(quote_agent.llm, "call_groq", fake)
        generate_quotes(n=1, pool_path=pool_file)
        generate_quotes(n=1, pool_path=pool_file)
        # The second request must mention the first quote.
        assert "The first entirely unique mock quote here." in captured[1]

    def test_figure_is_recorded_for_rotation(self, monkeypatch, pool_file):
        raw = json.dumps([{"quote": "Sun Tzu walked into a tavern and ordered nothing.", "format":
                           "twisted_proverb", "source": "Sun Tzu"}])
        monkeypatch.setattr(quote_agent.llm, "call_groq", lambda *a, **k: raw)
        got = generate_quotes(n=1, pool_path=pool_file)
        assert got[0].figure == "sun tzu"
        assert json.load(open(pool_file))["figures"] == ["sun tzu"]

    def test_nothing_is_persisted_when_generation_fails(self, monkeypatch, pool_file):
        monkeypatch.setattr(quote_agent.llm, "call_groq",
                            lambda *a, **k: _reply("Self-check: no, this is a real joke now."))
        with pytest.raises(RuntimeError):
            generate_quotes(n=1, pool_path=pool_file)
        assert not quote_agent._load_pool(pool_file).quotes
