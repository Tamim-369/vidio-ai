"""Characterization tests for src/utils/text_helpers.py.

These cover the JSON salvage helpers the quote generator relies on: Groq
routinely wraps its candidate list in prose or a fenced block, and a wedged
stream truncates mid-array, so _loads_json has to dig the array out rather than
raise.
"""
from __future__ import annotations

import pytest

from src.utils import text_helpers as th


class TestLoadsJson:
    def test_plain_array(self):
        assert th._loads_json('["a", "b"]') == ["a", "b"]

    def test_plain_object(self):
        assert th._loads_json('{"k": 1}') == {"k": 1}

    def test_strips_code_fences(self):
        assert th._loads_json('```json\n["a"]\n```') == ["a"]

    def test_strips_bare_fences(self):
        assert th._loads_json('```\n["a"]\n```') == ["a"]

    def test_ignores_leading_prose(self):
        assert th._loads_json('Here you go: ["a"]') == ["a"]

    def test_salvages_truncated_array(self):
        # Model died mid-stream: keep the complete leading elements.
        assert th._loads_json('["a", "b", "c') == ["a", "b"]

    def test_extracts_embedded_object(self):
        assert th._loads_json('noise {"k": "v"} noise') == {"k": "v"}

    def test_empty_output_raises(self):
        with pytest.raises(ValueError):
            th._loads_json("")

    def test_none_output_raises(self):
        with pytest.raises(ValueError):
            th._loads_json(None)

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            th._loads_json("no json here at all")


class TestFindBracket:
    def test_finds_balanced_region(self):
        assert th._find_bracket("x [a,b] y", "[", "]") == "[a,b]"

    def test_nested_brackets(self):
        assert th._find_bracket("[[a]]", "[", "]") == "[[a]]"

    def test_absent_returns_empty(self):
        assert th._find_bracket("abc", "[", "]") == ""

    def test_unbalanced_returns_tail(self):
        assert th._find_bracket("[abc", "[", "]") == "[abc"

    def test_works_for_braces(self):
        assert th._find_bracket('{"k": [1]}', "{", "}") == '{"k": [1]}'


class TestSalvageArray:
    def test_returns_none_when_nothing_complete(self):
        assert th._salvage_array("[{oops") is None

    def test_keeps_complete_prefix(self):
        assert th._salvage_array('[{"a":1},{"b":') == [{"a": 1}]

    def test_closed_array(self):
        assert th._salvage_array('[1,2,3]') == [1, 2, 3]
