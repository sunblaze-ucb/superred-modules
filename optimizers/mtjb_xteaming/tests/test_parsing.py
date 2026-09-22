"""Unit tests for the X-Teaming parsers (synthetic neutral strings only)."""

from __future__ import annotations

import json

from mtjb_xteaming_optimizer import parsing

_STRATEGY = {
    "persona": "p",
    "context": "c",
    "approach": "a",
    "conversation_plan": {"turn_1": "x", "final_turn": "y"},
}


def test_parse_strategies_returns_values() -> None:
    raw = json.dumps({f"S{i}": _STRATEGY for i in range(10)})
    strategies = parsing.parse_strategies(raw)
    assert strategies is not None and len(strategies) == 10


def test_parse_strategies_unwraps_code_fence() -> None:
    raw = "```json\n" + json.dumps({"S": _STRATEGY}) + "\n```"
    assert parsing.parse_strategies(raw) is not None


def test_parse_strategies_none_on_garbage() -> None:
    assert parsing.parse_strategies("not json") is None
    assert parsing.parse_strategies("") is None


def test_strategy_is_valid() -> None:
    assert parsing.strategy_is_valid(_STRATEGY) is True
    assert parsing.strategy_is_valid({"persona": "p"}) is False
    empty_plan = {"persona": "p", "context": "c", "approach": "a", "conversation_plan": {}}
    assert parsing.strategy_is_valid(empty_plan) is False


def test_extract_conversation_between_tags() -> None:
    assert parsing.extract_conversation("pre <conversation>HELLO</conversation> post") == "HELLO"


def test_extract_conversation_falls_back_without_tags() -> None:
    assert parsing.extract_conversation("  bare text  ") == "bare text"


def test_parse_judge_valid_range() -> None:
    assert parsing.parse_judge(json.dumps({"score": 5, "reason": "r"})) == (5, "r")
    assert parsing.parse_judge(json.dumps({"score": 1})) == (1, "")


def test_parse_judge_rejects_out_of_range() -> None:
    assert parsing.parse_judge(json.dumps({"score": 0})) is None
    assert parsing.parse_judge(json.dumps({"score": 6})) is None
    assert parsing.parse_judge("not json") is None
    assert parsing.parse_judge(json.dumps({"reason": "no score"})) is None


def test_format_strategy_includes_fields() -> None:
    rendered = parsing.format_strategy(_STRATEGY)
    assert "Persona:" in rendered and "Conversation Plan:" in rendered
