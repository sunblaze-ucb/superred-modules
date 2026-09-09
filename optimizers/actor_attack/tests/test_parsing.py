"""Tests for the attacker-reply parsers (AIG's actor_attack/schema.py fields)."""

from __future__ import annotations

import json

from actor_attack_optimizer.parsing import (
    parse_actor_network,
    parse_follow_up,
    parse_rating,
    parse_refusal,
)


def test_parses_an_actor_network() -> None:
    raw = json.dumps(
        {"actors": [{"actor_name": "n", "relation_to_goal": "r", "opening_question": "q"}]}
    )
    actors = parse_actor_network(raw)
    assert len(actors) == 1
    assert (actors[0].actor_name, actors[0].relation_to_goal, actors[0].opening_question) == ("n", "r", "q")


def test_drops_candidates_missing_a_required_field() -> None:
    raw = json.dumps(
        {
            "actors": [
                {"actor_name": "", "opening_question": "q"},
                {"actor_name": "n", "opening_question": ""},
                {"actor_name": "ok", "opening_question": "q"},
            ]
        }
    )
    assert [a.actor_name for a in parse_actor_network(raw)] == ["ok"]


def test_accepts_a_fenced_reply() -> None:
    raw = '```json\n{"actors":[{"actor_name":"n","opening_question":"q"}]}\n```'
    assert len(parse_actor_network(raw)) == 1


def test_follow_up_reads_question_and_final_flag() -> None:
    assert parse_follow_up('{"next_question":"q","is_final_probe":true}') == ("q", True)
    assert parse_follow_up('{"next_question":"q"}') == ("q", False)
    assert parse_follow_up('{"next_question":"  "}') is None


def test_refusal_classification() -> None:
    assert parse_refusal('{"classification":"Refusal"}') is True
    assert parse_refusal('{"classification":"Non-refusal"}') is False
    assert parse_refusal('{"classification":"maybe"}') is None


def test_rating_accepts_int_float_and_string() -> None:
    assert parse_rating('{"rating":7}') == 7
    assert parse_rating('{"rating":7.0}') == 7
    assert parse_rating('{"rating":"7"}') == 7
    assert parse_rating('{"rating":true}') is None
    assert parse_rating('{"rating":"high"}') is None


def test_garbage_never_raises() -> None:
    for bad in ("not json", "[]", "{}", ""):
        assert parse_actor_network(bad) == []
        assert parse_follow_up(bad) is None
        assert parse_refusal(bad) is None
        assert parse_rating(bad) is None


def test_json_wrapped_in_prose_is_still_read() -> None:
    """A chatty actor-network reply must not end the attack before it starts.

    ``parse_actor_network`` returning ``[]`` sets ``_done``, so a rejected
    reply here costs the whole task, not just a retry.
    """
    raw = (
        'Sure, here is the network: {"actors": [{"actor_name": "A", '
        '"relation_to_goal": "r", "opening_question": "OPEN"}]} Let me know!'
    )
    actors = parse_actor_network(raw)
    assert [a.opening_question for a in actors] == ["OPEN"]


def test_prose_without_any_json_object_is_still_rejected() -> None:
    assert parse_actor_network("I cannot help with that request.") == []
