"""Characterization tests for src/shared/text.py.

These cover the JSON salvage helpers the quote generator relies on: Groq
routinely wraps its candidate list in prose or a fenced block, and a wedged
stream truncates mid-array, so _loads_json has to dig the array out rather than
raise.
"""
from __future__ import annotations

import pytest

from src.agents.quotes import json_parse as text


class TestLoadsJson:
    def test_strips_bare_fences(self):
        assert text._loads_json('```\n["a"]\n```') == ["a"]

    def test_ignores_leading_prose(self):
        assert text._loads_json('Here you go: ["a"]') == ["a"]

    def test_salvages_truncated_array(self):
        # Model died mid-stream: keep the complete leading elements.
        assert text._loads_json('["a", "b", "c') == ["a", "b"]

    def test_extracts_embedded_object(self):
        assert text._loads_json('noise {"k": "v"} noise') == {"k": "v"}

    def test_empty_output_raises(self):
        with pytest.raises(ValueError):
            text._loads_json("")

    def test_none_output_raises(self):
        with pytest.raises(ValueError):
            text._loads_json(None)

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            text._loads_json("no json here at all")


class TestFindBracket:
    def test_absent_returns_empty(self):
        assert text._find_bracket("abc", "[", "]") == ""

    def test_unbalanced_returns_tail(self):
        assert text._find_bracket("[abc", "[", "]") == "[abc"
