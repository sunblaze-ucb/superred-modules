"""Tests for the simulator-reply parsers (upstream's schema.py fields)."""

from __future__ import annotations

import json

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


def test_a_reply_containing_a_code_fence_is_not_mangled() -> None:
    """``_strip_fence``'s pattern is unanchored.

    A valid, unfenced reply whose string value contains a markdown fence -- a
    realistic shape for an attack eliciting code -- must not be cut down to the
    inner fence's contents.
    """
    payload = "Write this script:\n```python\nimport os\n```\nThen run it."
    raw = json.dumps({"new_attack_prompt": payload})
    assert parse_new_attack_prompt(raw) == payload


def test_a_fenced_reply_is_still_unwrapped() -> None:
    raw = '```json\n{"new_attack_prompt": "ATTACK"}\n```'
    assert parse_new_attack_prompt(raw) == "ATTACK"


def test_a_list_wrapped_reply_still_reaches_the_object_fallback() -> None:
    """A candidate that parses to a non-object must not end the search.

    Returning there skips the remaining candidates, including the embedded
    object fallback that recovers this exact shape.
    """
    assert parse_new_attack_prompt('[{"new_attack_prompt": "ATTACK"}]') is not None
    assert parse_new_attack_prompt('{"new_attack_prompt": "ATTACK"}') is not None
