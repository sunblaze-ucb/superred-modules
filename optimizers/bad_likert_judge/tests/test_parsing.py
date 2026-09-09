"""Tests for the simulator-reply parsers (upstream's schema.py fields)."""

from __future__ import annotations

from bad_likert_judge_optimizer.parsing import (
    parse_new_attack_prompt,
    parse_refined_example,
    parse_refusal,
    parse_score_3_example,
)


def test_parses_each_schema_field() -> None:
    assert parse_score_3_example('{"score_1_example":"a","score_3_example":"b"}') == "b"
    assert parse_refined_example('{"refined_example":"r"}') == "r"
    assert parse_new_attack_prompt('{"new_attack_prompt":"p"}') == "p"
    assert parse_refusal('{"refusal": true, "reasoning": "x"}') is True
    assert parse_refusal('{"refusal": false}') is False


def test_accepts_a_fenced_reply() -> None:
    assert parse_new_attack_prompt('```json\n{"new_attack_prompt":"p"}\n```') == "p"


def test_string_booleans_are_read() -> None:
    assert parse_refusal('{"refusal":"true"}') is True
    assert parse_refusal('{"refusal":"no"}') is False


def test_unparseable_returns_none_rather_than_raising() -> None:
    for bad in ("not json", "[]", "{}", '{"other":1}', ""):
        assert parse_score_3_example(bad) is None
        assert parse_new_attack_prompt(bad) is None
        assert parse_refusal(bad) is None


def test_blank_values_are_rejected() -> None:
    assert parse_score_3_example('{"score_3_example":"   "}') is None
    assert parse_new_attack_prompt('{"new_attack_prompt":""}') is None


def test_json_wrapped_in_prose_is_still_read() -> None:
    """A chatty reply must not cost a turn and a backtrack."""
    raw = 'Certainly! {"new_attack_prompt": "ATTACK"} — let me know if you need more.'
    assert parse_new_attack_prompt(raw) == "ATTACK"


def test_prose_without_any_json_object_is_still_rejected() -> None:
    assert parse_new_attack_prompt("I cannot help with that request.") is None
