"""Tests for the per-character prompt structure.

Each character narrates and writes jokes, so its prompt is a property of the
character. These tests cover the wiring: the right module is chosen, each
prompt's own length window is what gets enforced, and a prompt's text cannot be
edited without a test failing.
"""
from __future__ import annotations

import hashlib

import pytest

from src.agents.quotes import agent as qa
from src.agents.quotes.prompt import characters_with_prompts, get_prompt, _load
from src.agents.quotes import prompt_andru_tatte as andru_tatte
from src.agents.quotes import prompt_brolexander as brolexander
from src.agents.quotes import prompt_don_tzu as don_tzu
from src.agents.quotes import prompt_shared as shared
from src.agents.voice_cast.voices import VOICES, get_enabled_voices

CHARACTERS = [vid for vid, v in VOICES.items() if v.get("enabled")]


class TestRegistry:
    def test_no_character_prompt_takes_a_subject(self):
        # The user's call: the quotes are about the character. Identity is the
        # check that works -- a probe string cannot be used, because a prompt
        # may legitimately contain the same word in its own examples.
        for vid in characters_with_prompts():
            mod = get_prompt(vid)
            assert mod.build_prompt("Money", 4) == mod.build_prompt("", 4), \
                f"{vid} still interpolates a subject"
            assert "SUBJECT" not in mod.build_prompt("Money", 4), \
                f"{vid} still asks for a subject"

    def test_the_shared_fallback_still_takes_a_subject(self):
        # It has no persona, so a subject is all it has to go on.
        assert "SUBJECT: Money" in shared.build_prompt("Money", 4)

    def test_every_prompt_module_offers_the_same_interface(self):
        for vid in characters_with_prompts():
            mod = get_prompt(vid)
            assert callable(mod.build_prompt), f"{vid} has no build_prompt"
            assert isinstance(mod.MIN_CHARS, int) and mod.MIN_CHARS > 0
            assert isinstance(mod.MAX_CHARS, int) and mod.MAX_CHARS > mod.MIN_CHARS

    def test_an_unknown_character_falls_back_instead_of_raising(self):
        # A new character must not be able to break quote generation.
        assert get_prompt("brand-new-voice") is shared
        assert get_prompt("") is shared

    def test_every_enabled_character_can_generate(self):
        for vid in CHARACTERS:
            assert get_prompt(vid) is not None

    def test_no_piped_character_is_silently_borrowing_the_shared_one(self):
        # Anyone in rotation must have a prompt of their own, not the fallback.
        missing = [vid for vid in CHARACTERS if get_prompt(vid) is shared]
        assert missing == [], f"falling back to the shared prompt: {missing}"

    def test_the_registry_is_keyed_by_voice_id(self):
        for vid in characters_with_prompts():
            assert vid in VOICES, f"{vid} is not a registered voice"

    def test_borrowing_the_shared_prompt_says_so(self, monkeypatch, tmp_path, capsys):
        # Otherwise a character with no prompt of its own looks like it has one.
        good = "Recovery is growth, therefore rest as much as you lift, endlessly."
        monkeypatch.setattr(qa.llm, "call_groq",
                            lambda *a, **k: qa.json.dumps([{"quote": good}]))
        qa.generate_quotes(n=1, subject="Weight lifting",
                           pool_path=str(tmp_path / "p.json"),
                           character="brand-new-voice")
        assert "brand-new-voice has no prompt yet" in capsys.readouterr().out

    def test_a_character_with_its_own_prompt_is_not_flagged(self, monkeypatch, tmp_path, capsys):
        good = "The enemy must never know your plans, especially if you have none."
        monkeypatch.setattr(qa.llm, "call_groq",
                            lambda *a, **k: qa.json.dumps([{"quote": good}]))
        qa.generate_quotes(n=1, subject="War", pool_path=str(tmp_path / "p.json"),
                           character="donald-trump")
        assert "no prompt yet" not in capsys.readouterr().out


class TestDonTzuPrompt:
    # sha256 of don_tzu.build_prompt("SUBJECT_TOKEN", 42).
    FIDELITY_SHA256 = "159080850feb9f79092fe703d70961f5dfe55fc97620344b0d9eaec9feb74777"

    def test_the_prompt_text_is_locked(self):
        built = don_tzu.build_prompt("SUBJECT_TOKEN", 42).encode()
        digest = hashlib.sha256(built).hexdigest()
        assert digest == self.FIDELITY_SHA256, (
            "Don Tzu's prompt text changed; it is the user's specification, so "
            "any edit has to be deliberate"
        )

    def test_the_count_lands_in_the_user_input_block(self):
        built = don_tzu.build_prompt("Money", 4)
        assert built.rstrip().endswith("NUMBER: 4")
        assert "USER INPUT:" in built

    def test_no_placeholder_survives(self):
        built = don_tzu.build_prompt("Money", 4)
        assert "{" not in built and "}" not in built
        assert "SUBJECT_TOKEN" not in built

    def test_it_takes_no_subject(self):
        # The quotes are about him, not about a per-video topic, so a subject
        # must not leak in and retarget them.
        assert don_tzu.build_prompt("Money", 4) == don_tzu.build_prompt("", 4)
        assert "SUBJECT" not in don_tzu.build_prompt("Money", 4)

    def test_he_is_fixed_to_war_and_strategy(self):
        built = don_tzu.build_prompt("Money", 4)
        assert "The underlying wisdom must come from WAR AND STRATEGY" in built
        for gone in ("the SUBJECT provided by the user", "If the SUBJECT is:",
                     "WEIGHT LIFTING:", "PROGRAMMING:", "BUSINESS:", "DATING:",
                     "For ANY SUBJECT:"):
            assert gone not in built, f"{gone!r} still refers to a subject"
        for kept in ("strategy", "enemies", "battles", "preparation", "victory",
                     "retreat"):
            assert kept in built

    def test_nothing_is_appended_to_it(self):
        built = don_tzu.build_prompt("Money", 4)
        for extra in ("JSON array", "HARD LIMIT", "markdown fences"):
            assert extra not in built

    def test_the_prompt_states_its_own_length_limit(self):
        assert don_tzu.MAX_CHARS == 100
        built = don_tzu.build_prompt("Money", 4)
        assert "EVERY QUOTE MUST BE 100 CHARACTERS OR FEWER" in built
        assert "Prefer 40–85 characters." in built

    def test_every_section_of_the_prompt_survived(self):
        built = don_tzu.build_prompt("Money", 4)
        for section in ("CORE COMEDY FORMULA", "EXAMPLES OF THE EXACT STYLE",
                        "ABSOLUTE RULE: EVERY QUOTE MUST BE FUNNY",
                        "MAXIMUM LENGTH", "VOCABULARY", "NO RANDOM JOKES",
                        "NO PUNCHLINES", "DON TZU'S PERSONALITY",
                        "SOURCE OF WISDOM", "THE TWIST", "VARIETY",
                        "FINAL QUALITY CHECK", "OUTPUT"):
            assert section in built, f"section {section!r} is missing"

    def test_it_identifies_the_character(self):
        assert don_tzu.build_prompt("Money", 4).startswith("You are DON TZU.")

    def test_it_carries_the_target_style_examples(self):
        built = don_tzu.build_prompt("Money", 4)
        assert "at least look confident" in built
        assert "never march after lunch" in built


class TestAndruTattePrompt:
    # sha256 of andru_tatte.build_prompt("SUBJECT_TOKEN", 42).
    FIDELITY_SHA256 = "cb79f4ea533bb305ac069e5cbd9c7ce12840ffed14e28e3f37164144d5092be3"

    def test_the_prompt_text_is_locked(self):
        digest = hashlib.sha256(
            andru_tatte.build_prompt("SUBJECT_TOKEN", 42).encode()).hexdigest()
        assert digest == self.FIDELITY_SHA256, (
            "Andru Tatte's prompt text changed; it is the user's specification, "
            "so any edit has to be deliberate"
        )

    def test_the_count_lands_in_the_output_block(self):
        built = andru_tatte.build_prompt("Money", 4)
        assert "NUMBER: 4" in built
        # This prompt asks for the count a second time, in a full sentence.
        assert "Generate exactly 4 quotes." in built

    def test_it_takes_no_subject(self):
        # The quotes are about him, not about a per-video topic, so a subject
        # must not leak in and retarget them.
        assert andru_tatte.build_prompt("Money", 4) == andru_tatte.build_prompt("", 4)
        assert "SUBJECT" not in andru_tatte.build_prompt("Money", 4)

    def test_both_mentions_of_the_variables_are_filled(self):
        # A half-interpolated prompt would still carry {n} in the sentence.
        built = andru_tatte.build_prompt("Money", 4)
        assert "{" not in built and "}" not in built
        assert "SUBJECT_TOKEN" not in built

    def test_every_section_of_the_prompt_survived(self):
        built = andru_tatte.build_prompt("Money", 4)
        for section in ("CORE FORMULA", "Examples of the style",
                        "CHARACTER STYLE", "IMPORTANT HUMOR RULE",
                        "USE THESE TYPES OF TWISTS", "RELATABILITY IS IMPORTANT",
                        "LANGUAGE", "LENGTH", "QUALITY CONTROL", "OUTPUT"):
            assert section in built, f"section {section!r} is missing"

    def test_the_style_examples_survived(self):
        built = andru_tatte.build_prompt("Money", 4)
        for example in ("A true warrior never backs down from a fight until his mom calls him to eat.",
                        "Nothing is impossible in this world, as long as you are willing to give up.",
                        "A strong man faces every problem. A smart man avoids some of them.",
                        "Never give up on your dreams. Unless they require waking up early.",
                        "The road to success is long. That is why I recommend taking a bus."):
            assert example in built, f"example {example[:40]!r} is missing"

    def test_the_bad_and_good_pairs_survived(self):
        # The contrast is the whole lesson of this prompt, so both halves of
        # each pair have to still be there. The four extra BAD examples come
        # from the funny-gate section, which exists because the model kept
        # writing true-and-relatable observations that nobody laughs at.
        built = andru_tatte.build_prompt("Money", 4)
        assert built.count("BAD:") == 6
        assert built.count("GOOD:") == 5
        assert "Success requires hard work, which is why I respect people who work tomorrow." in built
        assert "Life is a journey, so sometimes the best decision is to stay home." in built

    def test_the_twist_list_survived(self):
        built = andru_tatte.build_prompt("Money", 4)
        twists = [
            "Taking motivational advice too literally",
            "Giving up while pretending it is wisdom",
            "Being lazy but explaining it like philosophy",
            "Choosing comfort over ambition",
            "Overconfidence",
            "Misunderstanding a common saying",
            "Finding a ridiculous loophole in motivational advice",
            "Reversing the expected lesson",
            "Making a selfish decision sound noble",
            "Using brutally honest human behavior as the wisdom",
        ]
        for twist in twists:
            assert twist in built, f"twist {twist!r} is missing"

    def test_the_funny_gate_survived(self):
        # This is what separates his output from a wall of life platitudes, so
        # it is worth failing loudly if it is ever trimmed.
        built = andru_tatte.build_prompt("Money", 4)
        assert "ABSOLUTE RULE: EVERY QUOTE MUST BE FUNNY" in built
        assert "CONCRETE DETAIL" in built
        assert "Relatable and true is not funny." in built
        assert "If you cannot picture the second half, it is not a joke." in built
        assert "Would someone actually laugh at" in built
        assert "Do not output borderline quotes." in built

    def test_every_section_of_the_prompt_survived(self):
        built = andru_tatte.build_prompt("Money", 4)
        for section in ("ABSOLUTE RULE: EVERY QUOTE MUST BE FUNNY",
                        "CONCRETE DETAIL"):
            assert section in built, f"section {section!r} is missing"

    def test_his_centre_is_life_and_self_improvement(self):
        built = andru_tatte.build_prompt("", 4)
        assert "Andru Tatte is about LIFE, SELF-IMPROVEMENT and HARD WORK." in built
        for theme in ("effort", "discipline", "ambition", "mindset", "consistency",
                      "grinding", "self-respect"):
            assert f"\n{theme}\n" in built, f"theme {theme!r} is missing"

    def test_he_is_not_narrowed_to_one_subject(self):
        # The user was explicit: no topic limit, and gym and war are fair game.
        built = andru_tatte.build_prompt("", 4)
        assert "He is NOT about one subject." in built
        assert "Do not narrow him down." in built
        for topic in ("the gym and weight lifting", "war and conflict"):
            assert topic in built, f"topic {topic!r} is missing"
        assert "Heavy lifts build discipline" in built
        assert "calm under the" in built, "no war example to pattern-match"

    def test_money_is_not_his_centre(self):
        # Money may still appear naturally, but nothing may push it to the front.
        built = andru_tatte.build_prompt("", 4)
        assert "penny" not in built
        assert "\nmoney\n" not in built
        assert "\nbeing broke\n" not in built

    def test_the_quality_control_questions_survived(self):
        built = andru_tatte.build_prompt("Money", 4)
        assert "Before outputting a quote, silently ask:" in built
        assert "6. Is it actually funny?" in built
        assert "If the answer to #6 is NO, DELETE IT and create another one." in built

    def test_the_banned_objects_list_survived(self):
        built = andru_tatte.build_prompt("Money", 4)
        assert "coffee, sandwiches, Wi-Fi, Tuesday, penguins, bananas, emails, etc." in built

    def test_the_prompt_states_its_own_length_limit(self):
        assert andru_tatte.MAX_CHARS == 100
        built = andru_tatte.build_prompt("Money", 4)
        assert "Maximum 100 characters per quote, including spaces and punctuation." in built
        assert "Prefer 40–90 characters." in built

    def test_it_is_registered_for_his_voice_id(self):
        assert get_prompt("andrew-tate") is andru_tatte


class TestBrolexanderPrompt:
    # sha256 of brolexander.build_prompt("SUBJECT_TOKEN", 42).
    FIDELITY_SHA256 = "ff2054d886346d874aeb1f21dee7030827c0cb2da8618bf45e25fb5ca573ac6a"

    def test_the_prompt_text_is_locked(self):
        digest = hashlib.sha256(
            brolexander.build_prompt("SUBJECT_TOKEN", 42).encode()).hexdigest()
        assert digest == self.FIDELITY_SHA256, (
            "Brolexander's prompt text changed; it is the user's specification, "
            "so any edit has to be deliberate"
        )

    def test_the_count_lands_in_both_places_it_is_asked_for(self):
        built = brolexander.build_prompt("", 4)
        assert "NUMBER: 4" in built
        assert "Generate exactly 4 quotes." in built

    def test_no_placeholder_survives(self):
        built = brolexander.build_prompt("", 4)
        assert "{" not in built and "}" not in built

    def test_it_takes_no_subject(self):
        # His gym framing is the topic, so a per-video subject must not leak in
        # and silently retarget the quotes.
        assert brolexander.build_prompt("Weight lifting and bodybuilding", 4) == \
            brolexander.build_prompt("", 4)
        assert "SUBJECT" not in brolexander.build_prompt("Money", 4)

    def test_every_section_of_the_prompt_survived(self):
        built = brolexander.build_prompt("", 4)
        for section in ("CHARACTER:", "CORE COMEDY FORMULA:", "Examples:",
                        "IMPORTANT:", "HUMOR RULE:", "GYM LOGIC TO USE:",
                        "VARIETY:", "LANGUAGE:", "LENGTH:", "QUALITY CONTROL:",
                        "NEVER output:", "OUTPUT:"):
            assert section in built, f"section {section!r} is missing"

    def test_the_gym_is_his_lens_survived(self):
        built = brolexander.build_prompt("", 4)
        for line in ("Every problem is a workout.",
                     "Every relationship is a training partner.",
                     "Every failure is progressive overload.",
                     "Every success is a PR.",
                     "Every argument is a set.",
                     "Every period of rest is recovery."):
            assert line in built, f"{line!r} is missing"

    def test_the_style_examples_survived(self):
        built = brolexander.build_prompt("", 4)
        for example in ("Life is about balance. That is why I train one arm at a time.",
                        "Money comes and go. Gains stay.",
                        "A broken heart is temporary. Leg day is forever.",
                        "Sleep is important. You cannot PR while unconscious.",
                        "My enemies are not my problem. They are my progressive overload."):
            assert example in built, f"example {example[:40]!r} is missing"

    def test_the_bad_and_good_contrast_survived(self):
        built = brolexander.build_prompt("", 4)
        # The two extra BAD examples are the "comparison plus explanation" form
        # the model kept falling into, added because it was never shown as wrong.
        # Counted as lines, not substrings, so a marker quoted mid-sentence in
        # the prose does not inflate the tally.
        # Counted as lines, not substrings, so a marker quoted mid-sentence in
        # the prose does not inflate the tally.
        lines = built.splitlines()
        assert lines.count("BAD:") == 9
        assert lines.count("GOOD:") == 6
        assert lines.count("WRONG:") == 4
        assert "Never run from your problems. Unless it is cardio day." in built
        assert "Love requires commitment. So does a 12-week bulk." in built
        assert "Love is a long set. You keep spotting each other until the reps never end." in built
        # His character is defined by "every X is a Y", which is exactly what
        # made him write "Love is a spotter. I lock my wrist to yours." Banning
        # the equating opening is what finally unlocked the jokes, so this rule
        # is load-bearing and must not be trimmed as redundant with the gate.
        assert "DO NOT BEGIN BY EQUATING TWO THINGS" in built
        assert "describe how he THINKS, not how he writes" in built
        assert 'WRONG:\n"Love is a spotter."' in built
        assert 'WRONG:\n"Sleep is recovery."' in built
        assert "RIGHT, plain truth first, absurd gym claim second:" in built

    def test_the_gym_logic_list_survived(self):
        built = brolexander.build_prompt("", 4)
        for term in ("progressive overload", "gains", "PRs", "sets", "reps",
                     "bulk", "cut", "protein", "leg day", "spotting", "cardio",
                     "pre-workout", "training", "rest"):
            assert f"* {term}" in built, f"gym term {term!r} is missing"

    def test_the_topic_list_survived(self):
        built = brolexander.build_prompt("", 4)
        for topic in ("love", "friendship", "money", "work", "school", "sleep",
                      "family", "aging", "laziness", "motivation"):
            assert f"\n{topic}\n" in built, f"topic {topic!r} is missing"

    def test_the_quality_control_questions_survived(self):
        built = brolexander.build_prompt("", 4)
        assert "Before outputting each quote, silently check:" in built
        assert "1. Does this sound like Brolexander?" in built
        assert "7. Is it under 100 characters?" in built

    def test_the_prompt_states_its_own_length_limit(self):
        assert brolexander.MAX_CHARS == 100
        built = brolexander.build_prompt("", 4)
        assert "Maximum 100 characters per quote, including spaces and punctuation." in built
        assert "Prefer 40–90 characters." in built

    def test_he_is_back_in_the_piped_characters(self):
        # He was held out while this prompt was reworked, and is back now.
        assert "arnold-schwarzenegger" in characters_with_prompts()
        assert get_prompt("arnold-schwarzenegger") is brolexander

    def test_his_funny_gate_survived(self):
        # The same two rules that fixed Andru Tatte are what stopped Brolexander
        # writing gym metaphors with sincere explanations.
        built = brolexander.build_prompt("", 4)
        assert "ABSOLUTE RULE: EVERY QUOTE MUST BE FUNNY" in built
        assert "CONCRETE DETAIL" in built
        assert "If you cannot picture the second half, it is not a joke." in built
        assert "Would someone actually laugh at" in built
        assert "Do not output borderline quotes." in built
        # His own failure mode has to stay shown as wrong, or he drifts back.
        assert "Love is a long set. You keep spotting each other until the reps never end." in built


class TestLengthWindowFollowsTheCharacter:
    def test_don_tzu_quotes_are_capped_at_100(self, monkeypatch, tmp_path):
        pool = str(tmp_path / "pool.json")
        long_quote = "x" * 140
        monkeypatch.setattr(qa.llm, "call_groq",
                            lambda *a, **k: qa.json.dumps([{"quote": long_quote}]))
        with pytest.raises(RuntimeError, match="usable quotes"):
            qa.generate_quotes(n=1, subject="War", pool_path=pool,
                               character="donald-trump")

    def test_don_tzu_accepts_a_quote_inside_his_window(self, monkeypatch, tmp_path):
        pool = str(tmp_path / "pool.json")
        good = "The enemy must never know your plans, especially if you have none."
        monkeypatch.setattr(qa.llm, "call_groq",
                            lambda *a, **k: qa.json.dumps([{"quote": good}]))
        got = qa.generate_quotes(n=1, subject="War", pool_path=pool,
                                 character="donald-trump")
        assert got[0].text == good
        assert len(good) <= don_tzu.MAX_CHARS

    def test_a_character_without_a_prompt_keeps_the_shared_window(self, monkeypatch, tmp_path):
        # 140 chars is rejected for Don Tzu but accepted under the shared prompt,
        # which is what lets his 100-char cap stay his own rule.
        pool = str(tmp_path / "pool.json")
        long_quote = "x" * 140
        monkeypatch.setattr(qa.llm, "call_groq",
                            lambda *a, **k: qa.json.dumps([{"quote": long_quote}]))
        got = qa.generate_quotes(n=1, subject="Sleep", pool_path=pool,
                                 character="brand-new-voice")
        assert got[0].text == long_quote


class TestAgentUsesTheCharactersPrompt:
    def test_the_characters_prompt_is_the_one_sent(self, monkeypatch, tmp_path):
        captured = []
        monkeypatch.setattr(qa.llm, "call_groq",
                            lambda messages, **k: (captured.append(messages[0]["content"]),
                                                   qa.json.dumps([{"quote": "Know your enemy and know yourself, then look confident."}]))[1])
        qa.generate_quotes(n=1, subject="War", pool_path=str(tmp_path / "p.json"),
                           character="donald-trump")
        assert captured[0].startswith(don_tzu.build_prompt("War", qa.BATCH)[:40])
        assert "You are DON TZU." in captured[0]

    def test_another_character_gets_a_different_prompt(self, monkeypatch, tmp_path):
        captured = []
        good = "Recovery is growth, therefore rest as much as you lift, endlessly."
        monkeypatch.setattr(qa.llm, "call_groq",
                            lambda messages, **k: (captured.append(messages[0]["content"]),
                                                   qa.json.dumps([{"quote": good}]))[1])
        qa.generate_quotes(n=1, subject="Weight lifting",
                           pool_path=str(tmp_path / "p.json"),
                           character="arnold-schwarzenegger")
        assert "You are DON TZU." not in captured[0]
        assert captured[0] != don_tzu.build_prompt("Weight lifting", qa.BATCH)
