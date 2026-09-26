"""Tests for the quote agent.

The network path is stubbed: every test here exercises parsing, filtering,
rotation and persistence against a faked llm.call_groq, so the suite costs no
Groq calls and runs offline.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from src.agents.quotes import agent as quote_agent
from src.agents.quotes import prompt_shared as prompt_mod
from src.agents.quotes.agent import Quote, _Pool, _build_messages, generate_quotes


@pytest.fixture
def pool_file(tmp_path):
    return str(tmp_path / "used_quotes.json")


def _reply(*quotes):
    """Build a model reply in the requested JSON array format."""
    return json.dumps([{"quote": q} for q in quotes])


# --- parsing -----------------------------------------------------------------

class TestParseCandidates:
    def test_parses_requested_json_array(self):
        got = quote_agent.parse_candidates(_reply("The very first mock quote in this batch.", "The second mock quote in this batch here."))
        assert [c["quote"] for c in got] == ["The very first mock quote in this batch.", "The second mock quote in this batch here."]

    def test_reads_bare_array(self):
        assert quote_agent.parse_candidates('["A bare string mock quote, not an object."]')[0]["quote"] == \
            "A bare string mock quote, not an object."

    def test_normalises_the_hyphen_the_model_gets_wrong(self):
        # The model emits U+2011, which TTS reads as one word and the card font
        # may not carry, so it has to become a plain hyphen before either sees it.
        got = quote_agent.parse_candidates(
            "Invest in yourself; if you can't, invest in a couch and call it a long‑term asset.")[0]["quote"]
        assert got == "Invest in yourself; if you can't, invest in a couch and call it a long-term asset."
        assert "‑" not in got

    def test_normalises_lookalike_spaces_and_strips_zero_width(self):
        got = quote_agent._strip_wrapper("A man who disciplines himself owns nothing.")
        assert got == "A man who disciplines himself owns nothing."

    def test_keeps_typography_that_is_not_a_confusable(self):
        # A curly apostrophe is real punctuation and both TTS and the font
        # handle it, so it must survive untouched.
        got = quote_agent._strip_wrapper("Money doesn’t talk, it simply screams quietly.")
        assert "doesn’t" in got

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

    def test_ignores_keys_the_prompt_never_asks_for(self):
        # The joke prompt requests only a quote, but a model may volunteer more.
        raw = '[{"quote": "A short mock quote for the parser tests.", "format": "one_liner"}]'
        got = quote_agent.parse_candidates(raw)
        assert got[0] == {"quote": "A short mock quote for the parser tests."}

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

    def test_the_length_window_is_30_to_200(self):
        # The prompt asks for "preferably 10-35 words"; 35 words is ~200 chars.
        # The old 80-char cap rejected almost every joke the new prompt produced.
        assert quote_agent.MIN_QUOTE_CHARS == 30
        assert quote_agent.MAX_QUOTE_CHARS == 200

    def test_a_200_char_quote_is_accepted(self):
        text = "x" * 200
        assert quote_agent.reject_reason({"quote": text}, _Pool()) == ""

    def test_201_chars_is_rejected(self):
        text = "x" * 201
        assert quote_agent.reject_reason({"quote": text}, _Pool()) == "too long"

    def test_a_real_170_char_joke_fits(self):
        # Length taken from an actual quote the new prompt produced on a live run.
        text = ("When you surround an army, leave an outlet open. If the army does not "
                "use the outlet, you have validated your intelligence on their retreat patterns.")
        assert 30 <= len(text) <= 200
        assert quote_agent.reject_reason({"quote": text}, _Pool()) == ""

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
        assert quote_agent.reject_reason({"quote": text + "x" * 136}, _Pool()) == "too long"

    def test_the_prompt_states_its_own_length_guidance(self):
        # The bounds are the agent's card budget, not the prompt's wording, so
        # the content has to be the prompt this character actually uses.
        content = _build_messages(_Pool(), 2, "Money", prompt_mod)[0]["content"]
        assert "10–35 words" in content

    def test_rejects_duplicate_regardless_of_case_and_punctuation(self):
        pool = _Pool(quotes=[{"text": "The tank politely exploded, again, on schedule."}])
        cand = {"quote": "the tank politely exploded, again, on schedule"}
        assert quote_agent.reject_reason(cand, pool) == "already used"

    def test_rejects_duplicate_ignoring_extra_spacing(self):
        pool = _Pool(quotes=[{"text": "The tank politely exploded, again, on schedule."}])
        assert quote_agent.reject_reason({"quote": "The  tank   politely exploded, again, on schedule."}, pool) == \
            "already used"

    def test_ignores_a_volunteered_format_key(self):
        # The joke prompt asks only for a quote, so a format tag is not a
        # rejection reason the way it was under the old anti-wisdom prompt.
        cand = {"quote": "A perfectly fine quote here, give or take.", "format": "haiku"}
        assert quote_agent.reject_reason(cand, _Pool()) == ""


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


# --- pool persistence --------------------------------------------------------

class TestPool:
    def test_missing_file_gives_empty_pool(self, pool_file):
        pool = quote_agent._load_pool(pool_file)
        assert pool.quotes == []

    def test_corrupt_file_gives_empty_pool(self, pool_file):
        with open(pool_file, "w") as f:
            f.write("{not json")
        assert quote_agent._load_pool(pool_file).quotes == []

    def test_round_trip(self, pool_file):
        pool = _Pool()
        quote_agent._remember(pool, Quote(text="A remembered mock quote from the old pool."))
        quote_agent._save_pool(pool, pool_file)
        back = quote_agent._load_pool(pool_file)
        assert back.quotes[0]["text"] == "A remembered mock quote from the old pool."

    def test_creates_parent_directory(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "pool.json")
        quote_agent._save_pool(_Pool(), path)
        assert (tmp_path / "deep" / "nested" / "pool.json").is_file()

    def test_state_file_is_inside_src_by_default(self):
        # Keeps the pool durable and out of the repo root.
        assert quote_agent.STATE_FILE.startswith("src/")


class TestPromptFidelity:
    """The prompt is the user's specification and is sent unmodified.

    The text is the user's with only SUBJECT and NUMBER filled in. Nothing is
    appended and no rule is restated, so the fidelity lock is a hash of the
    built prompt: any edit to the wording, any added instruction, or a dropped
    line changes it and fails here.
    """

    # sha256 of build_prompt("SUBJECT_TOKEN", 42). Update deliberately, never
    # to make a failing test go green.
    FIDELITY_SHA256 = "dae70b8034fb50aa71fd106722768bb292be184d4106a48ad1dfd95afe20b81a"

    def test_the_prompt_matches_the_original_exactly(self):
        built = prompt_mod.build_prompt("SUBJECT_TOKEN", 42).encode()
        assert hashlib.sha256(built).hexdigest() == self.FIDELITY_SHA256, (
            "the prompt text changed; it is the user's specification, so any "
            "edit has to be deliberate"
        )

    def test_it_is_a_function_of_subject_and_count(self):
        assert callable(prompt_mod.build_prompt)
        assert not hasattr(prompt_mod, "JOKE_PROMPT"), (
            "the prompt should be built by a function, not a constant"
        )

    def test_the_subject_lands_in_the_current_request_block(self):
        built = prompt_mod.build_prompt("Money", 3)
        assert "SUBJECT: Money" in built
        # The example list must stay generic, not be rewritten per subject.
        assert "SUBJECT = WAR" in built

    def test_the_count_lands_in_the_current_request_block(self):
        built = prompt_mod.build_prompt("Money", 3)
        assert "NUMBER: 3" in built
        assert "NUMBER: [number]" in built, "the prompt's own template line is part of the text"

    def test_no_placeholder_token_survives(self):
        built = prompt_mod.build_prompt("Money", 3)
        assert "[INSERT" not in built
        assert "[something]" in built and "[number]" in built, (
            "the prompt's literal template examples are part of its text"
        )

    def test_no_uninterpolated_brace_survives(self):
        built = prompt_mod.build_prompt("Money", 3)
        assert "{" not in built and "}" not in built

    def test_nothing_is_appended(self):
        built = prompt_mod.build_prompt("Money", 3)
        for extra in ("JSON array", "HARD LIMIT", "Return 3 quotes as a JSON",
                      "markdown fences", "between 30 and 200 characters"):
            assert extra not in built, f"{extra!r} is not the user's wording"

    def test_the_prompt_still_states_its_own_rules(self):
        built = prompt_mod.build_prompt("Money", 3)
        for line in ("Output ONLY the quotes.",
                     "10–35 words",
                     "Use one primary mechanism per quote.",
                     "Do not output the analysis.",
                     "MAKE WISDOM WRONG IN AN INTERESTING WAY."):
            assert line in built

    def test_the_prompt_forbids_the_jokes_it_is_avoiding(self):
        # The anti-punchline and no-random-objects rules are the whole point of
        # the current wording, so a trim to either should fail here.
        built = prompt_mod.build_prompt("Money", 3)
        assert "NO PUNCHLINE LANGUAGE" in built
        assert "NO RANDOM FUNNY OBJECTS" in built
        assert "DO NOT MAKE WISDOM FUNNY." in built

    @pytest.mark.parametrize("subject,count", [
        ("Money", 1), ("Weight lifting and bodybuilding", 6), ("Sleep", 12),
    ])
    def test_interpolation_scales(self, subject, count):
        built = prompt_mod.build_prompt(subject, count)
        assert f"SUBJECT: {subject}" in built
        assert f"NUMBER: {count}" in built


class TestPromptIntegrity:
    """Spot-checks on the wording that defines the joke mechanism."""

    VERBATIM_LINES = [
        "You are a specialized generator of **fake wisdom quotes**.",
        "The humor must come from **corrupting genuine wisdom**, not from adding punchlines.",
        "ARRIVE AT AN ABSURD BUT STRANGELY LOGICAL CONCLUSION",
        "Do NOT write a normal joke and disguise it as a quotation.",
        "Use one primary mechanism per quote.",
        "They are NOT intentionally telling a joke.",
        "The quote itself must be funny.",
    ]

    QUALITY_FILTER_LINES = [
        "it has an obvious punchline",
        "it introduces a random funny object",
        "it does not contain genuine wisdom from the SUBJECT",
        "it is random nonsense",
    ]

    @pytest.mark.parametrize("line", VERBATIM_LINES)
    def test_rule_line_present_verbatim(self, line):
        assert line in prompt_mod.build_prompt("Money", 3)

    @pytest.mark.parametrize("line", QUALITY_FILTER_LINES)
    def test_quality_filter_present_verbatim(self, line):
        assert line in prompt_mod.build_prompt("Money", 3)

    @pytest.mark.parametrize("mutation", [
        "FALSE DEDUCTION", "OVEREXTENSION", "LITERAL INTERPRETATION",
        "CONFIDENT MISUNDERSTANDING", "SELF-DEFEATING LOGIC", "ABSURD REDEFINITION",
        "WRONG PRIORITY", "UNEXPECTED CONSEQUENCE", "PHILOSOPHICAL PARADOX",
        "CONFIDENT IGNORANCE",
    ])
    def test_every_mutation_is_listed(self, mutation):
        assert mutation in prompt_mod.build_prompt("Money", 3)

    def test_the_prompt_refuses_to_assume_one_subject(self):
        built = prompt_mod.build_prompt("Money", 3)
        assert "DO NOT assume the subject is always war." in built
        assert "The SUBJECT can be absolutely anything." in built


# --- generation (stubbed network) -------------------------------------------

class TestGenerateQuotes:
    def test_returns_requested_count(self, monkeypatch, pool_file):
        monkeypatch.setattr(
            quote_agent.llm, "call_groq",
            lambda *a, **k: _reply("The first valid mock quote in this batch.", "The second valid mock quote in this batch."))
        got = generate_quotes(n=2, pool_path=pool_file)
        assert len(got) == 2
        assert all(isinstance(q, Quote) for q in got)

    def test_the_subject_reaches_the_prompt(self, monkeypatch, pool_file):
        # The subject is what makes a joke about the character's topic, so it
        # must be substituted into the user's prompt before the call, not just
        # printed or used for the filename.
        captured = []

        def fake(messages, **k):
            captured.append(messages[0]["content"])
            return _reply("A perfectly valid mock quote right here.")

        monkeypatch.setattr(quote_agent.llm, "call_groq", fake)
        generate_quotes(n=1, subject="Weight lifting and bodybuilding", pool_path=pool_file)
        prompt = captured[0]
        assert "SUBJECT: Weight lifting and bodybuilding" in prompt
        assert "[INSERT SUBJECT HERE]" not in prompt

    def test_different_characters_ask_for_different_subjects(self, monkeypatch, pool_file):
        # Proves the rotation requirement end to end: a different character
        # means the prompt is about something else entirely.
        seen = []
        replies = iter([
            _reply("Strategy is simply knowing the field before anyone else does."),
            _reply("Wealth is only weight that happens to sit in a bank."),
        ])

        def fake(messages, **k):
            seen.append(messages[0]["content"])
            return next(replies)

        monkeypatch.setattr(quote_agent.llm, "call_groq", fake)
        for subject in ("War and military strategy", "Money"):
            generate_quotes(n=1, subject=subject, pool_path=pool_file)
        assert "SUBJECT: War and military strategy" in seen[0]
        assert "SUBJECT: Money" in seen[1]
        assert seen[0] != seen[1]

    def test_dropped_candidates_are_silent_by_default(self, monkeypatch, pool_file, capsys):
        monkeypatch.setattr(quote_agent.llm, "call_groq",
                            lambda *a, **k: _reply("short"))
        generate_quotes(n=1, pool_path=pool_file) if False else None
        # "short" is under the floor, so nothing is accepted; a normal run must
        # not print the reason, or every production render gets debug noise.
        try:
            generate_quotes(n=1, pool_path=pool_file)
        except RuntimeError:
            pass
        assert "dropped" not in capsys.readouterr().out

    def test_explain_reports_each_dropped_candidate(self, monkeypatch, pool_file, capsys):
        monkeypatch.setattr(quote_agent.llm, "call_groq",
                            lambda *a, **k: _reply("short", "Also far too short to keep."))
        try:
            generate_quotes(n=1, pool_path=pool_file, explain=True)
        except RuntimeError:
            pass
        out = capsys.readouterr().out
        assert "dropped (too short)" in out
        assert "Also far too short to keep." in out

    def test_explain_is_wired_to_script_only(self, monkeypatch, pool_file, capsys):
        monkeypatch.setattr(quote_agent.llm, "call_groq",
                            lambda *a, **k: _reply("short", "Also far too short to keep."))
        try:
            generate_quotes(n=1, pool_path=pool_file, explain=False)
        except RuntimeError:
            pass
        assert "dropped" not in capsys.readouterr().out

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

    def test_only_the_prompt_is_sent_when_nothing_has_been_used(self, monkeypatch, pool_file):
        captured = []
        monkeypatch.setattr(quote_agent.llm, "call_groq",
                            lambda messages, **k: (captured.append(messages[0]["content"]),
                                                   _reply("A perfectly valid mock quote here."))[1])
        generate_quotes(n=1, subject="Money", pool_path=pool_file)
        # With an empty history the sent message is the prompt, byte for byte.
        assert captured[0] == prompt_mod.build_prompt("Money", quote_agent.BATCH)

    def test_nothing_is_appended_to_the_prompt(self, monkeypatch, pool_file):
        captured = []
        monkeypatch.setattr(quote_agent.llm, "call_groq",
                            lambda messages, **k: (captured.append(messages[0]["content"]),
                                                   _reply("A perfectly valid mock quote here."))[1])
        generate_quotes(n=1, subject="Money", pool_path=pool_file)
        sent = captured[0]
        assert sent.startswith(prompt_mod.build_prompt("Money", quote_agent.BATCH))
        for extra in ("HARD LIMIT", "These were rejected", "JSON array"):
            assert extra not in sent, f"{extra!r} must not be added to the user's prompt"

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

    def test_nothing_is_persisted_when_generation_fails(self, monkeypatch, pool_file):
        monkeypatch.setattr(quote_agent.llm, "call_groq",
                            lambda *a, **k: _reply("Self-check: no, this is a real joke now."))
        with pytest.raises(RuntimeError):
            generate_quotes(n=1, pool_path=pool_file)
        assert not quote_agent._load_pool(pool_file).quotes
