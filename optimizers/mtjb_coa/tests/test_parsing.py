"""Unit tests for the CoA response parsers (synthetic neutral strings only)."""

from __future__ import annotations

import json

from mtjb_coa_optimizer import parsing


def test_extract_chain_from_json_object() -> None:
    raw = json.dumps(
        {
            "1": {"improvement": "i1", "prompt": "PROMPT_ONE"},
            "2": {"improvement": "i2", "prompt": "PROMPT_TWO"},
            "evaluation": [{"score": 3}],
        }
    )
    assert parsing.extract_chain(raw) == ["PROMPT_ONE", "PROMPT_TWO"]


def test_extract_chain_ignores_prose_around_json() -> None:
    raw = "here is the chain:\n" + json.dumps({"1": {"improvement": "x", "prompt": "P"}})
    assert parsing.extract_chain(raw) == ["P"]


def test_extract_chain_none_on_garbage() -> None:
    assert parsing.extract_chain("no json here") is None
    assert parsing.extract_chain("") is None


def test_extract_chain_none_when_prompt_missing() -> None:
    assert parsing.extract_chain(json.dumps({"1": {"improvement": "x"}})) is None


def test_extract_update_prompt() -> None:
    raw = json.dumps({"improvement": "better", "prompt": "REFINED_PROMPT"})
    assert parsing.extract_update_prompt(raw) == "REFINED_PROMPT"


def test_extract_update_prompt_requires_both_keys() -> None:
    assert parsing.extract_update_prompt(json.dumps({"prompt": "x"})) is None


def test_clean_prompt_unwraps_list_syntax() -> None:
    assert parsing.clean_prompt("['wrapped']") == "wrapped"
    assert parsing.clean_prompt(["first", "second"]) == "first"
    assert parsing.clean_prompt("  plain  ") == "plain"


def test_extract_judge_score_binary() -> None:
    assert parsing.extract_judge_score("verdict [[0]]") == 0
    assert parsing.extract_judge_score("verdict [[1]]") == 1
    # any n is clamped to at most 1 (upstream behaviour)
    assert parsing.extract_judge_score("[[7]]") == 1
    assert parsing.extract_judge_score("no marker") is None


def test_extract_similarity_clamped() -> None:
    assert parsing.extract_similarity('{"similarity": 0.42}') == 0.42
    assert parsing.extract_similarity('{"similarity": 5.0}') == 1.0
    assert parsing.extract_similarity('{"similarity": -3}') == 0.0
    assert parsing.extract_similarity("not json") is None
    assert parsing.extract_similarity('{"other": 1}') is None
