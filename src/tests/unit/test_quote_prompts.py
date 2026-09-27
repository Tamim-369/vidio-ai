"""Tests for per-character prompt wiring.

Each character has its own prompt module, so a mistake here is silent: the
wrong prompt means the wrong character's jokes, with nothing crashing. What is
guarded is the wiring -- that every character resolves to a usable module, that
an unknown one falls back instead of raising, and that the module the character
owns is the one that reaches the model. The prompt *text* is deliberately not
asserted; it is meant to change, and pinning it would break on every edit
while preventing no bug.
"""
from __future__ import annotations

from src.agents.quotes import agent as qa
from src.agents.quotes.prompt import characters_with_prompts, get_prompt
from src.agents.quotes import prompt_don_tzu as don_tzu
from src.agents.quotes import prompt_shared as shared
from src.agents.voice_cast.voices import VOICES


class TestRegistry:
    def test_every_prompt_module_offers_the_same_interface(self):
        for vid in characters_with_prompts():
            mod = get_prompt(vid)
            assert callable(mod.build_prompt), f"{vid} has no build_prompt"
            assert isinstance(mod.MIN_CHARS, int) and mod.MIN_CHARS > 0
            assert isinstance(mod.MAX_CHARS, int) and mod.MAX_CHARS > mod.MIN_CHARS

    def test_an_unknown_character_falls_back_instead_of_raising(self):
        # A newly added voice must not be able to break quote generation.
        assert get_prompt("brand-new-voice") is shared
        assert get_prompt("") is shared

    def test_the_registry_is_keyed_by_voice_id(self):
        for vid in characters_with_prompts():
            assert vid in VOICES, f"{vid} is not a registered voice"


def test_the_characters_own_prompt_is_the_one_sent(monkeypatch, tmp_path):
    """A character must get its own prompt, not the shared fallback.

    The failure is silent -- quotes come back, just in the wrong voice -- so
    this checks what actually reached the model rather than what get_prompt
    returns.
    """
    captured = []

    def fake_call_groq(messages, **kwargs):
        captured.append(messages[0]["content"])
        return qa.json.dumps([{"quote": "Know your enemy, then look confident."}])

    monkeypatch.setattr(qa.llm, "call_groq", fake_call_groq)
    qa.generate_quotes(n=1, subject="War", pool_path=str(tmp_path / "p.json"),
                       character="donald-trump")

    assert captured, "no request reached the model"
    expected = don_tzu.build_prompt("War", qa.BATCH)
    assert captured[0].startswith(expected[:40]), "donald-trump got someone else's prompt"
