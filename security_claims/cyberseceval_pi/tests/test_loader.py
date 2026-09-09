"""Tests for the CyberSecEval prompt-injection loader."""

from __future__ import annotations

import json
from importlib import resources

import pytest
from cyberseceval_pi_claim.loader import DATASET_NAMES, load_cases


def test_dataset_names() -> None:
    assert DATASET_NAMES == ("english", "multilingual")


def test_english_has_251_cases() -> None:
    assert len(load_cases(dataset="english")) == 251


def test_multilingual_has_1004_cases() -> None:
    assert len(load_cases(dataset="multilingual")) == 1004


def test_cases_carry_required_fields() -> None:
    case = load_cases(dataset="english", limit=1)[0]
    assert case.system_prompt and case.user_input and case.judge_question


def test_fields_map_from_upstream_columns() -> None:
    raw = json.loads(
        resources.files("cyberseceval_pi_claim.data")
        .joinpath("prompt_injection.json")
        .read_text("utf-8")
    )
    case = load_cases(dataset="english", limit=1)[0]
    assert case.system_prompt == raw[0]["test_case_prompt"]
    assert case.user_input == raw[0]["user_input"]
    assert case.judge_question == raw[0]["judge_question"]


def test_filter_by_injection_type() -> None:
    cases = load_cases(dataset="english", injection_types=["direct"])
    assert cases
    assert all(c.injection_type == "direct" for c in cases)


def test_filter_by_risk_category() -> None:
    cases = load_cases(dataset="english", risk_categories=["security-violating"])
    assert cases
    assert all(c.risk_category == "security-violating" for c in cases)


def test_limit_caps_in_order() -> None:
    cases = load_cases(dataset="english", limit=10)
    assert len(cases) == 10
    assert [c.prompt_id for c in cases] == [
        c.prompt_id for c in load_cases(dataset="english")[:10]
    ]


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        load_cases(dataset="klingon")


def test_unmatched_filter_raises() -> None:
    with pytest.raises(ValueError, match="no cases matched"):
        load_cases(dataset="english", injection_types=["nonesuch"])


def test_bad_limit_raises() -> None:
    with pytest.raises(ValueError, match="limit must be"):
        load_cases(dataset="english", limit=0)
