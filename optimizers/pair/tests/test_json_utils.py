"""Tests for official-style PAIR JSON parsing."""

from __future__ import annotations

import pytest

from pair_optimizer.json_utils import JsonExtractionError, extract_attack_json


def test_extracts_first_json_object_and_removes_newlines() -> None:
    proposal = extract_attack_json(
        'prefix {\n"improvement": "better", "prompt": "attack"\n} suffix'
    )
    assert proposal.improvement == "better"
    assert proposal.prompt == "attack"
    assert proposal.system_prompt is None


def test_system_prompt_is_optional_extension() -> None:
    proposal = extract_attack_json(
        '{"improvement": "better", "prompt": "attack", "system_prompt": "override"}'
    )
    assert proposal.system_prompt == "override"


def test_missing_prompt_or_improvement_fails() -> None:
    with pytest.raises(JsonExtractionError):
        extract_attack_json('{"prompt": "attack"}')


def test_non_string_system_prompt_fails_when_present() -> None:
    with pytest.raises(JsonExtractionError):
        extract_attack_json('{"improvement": "better", "prompt": "attack", "system_prompt": 3}')
