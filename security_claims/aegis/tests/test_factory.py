"""Factory: task counts, filters, and construction (offline judge)."""

from __future__ import annotations

import pytest
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.types.llm import LLMConfig

from aegis_claim.factory import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    UNSAFE_COUNT_HARM_ONLY,
    UNSAFE_COUNT_WITH_NEEDS_CAUTION,
    aegis_claim,
)
from aegis_claim.judge import RefusalHeuristicJudge
from aegis_claim.task import AegisTask


def _claim(**kwargs: object) -> SecurityClaim:
    return aegis_claim(judge=RefusalHeuristicJudge(), **kwargs)  # type: ignore[arg-type]


def test_default_claim_has_236_tasks() -> None:
    tasks = list(_claim())
    assert len(tasks) == 236
    assert UNSAFE_COUNT_WITH_NEEDS_CAUTION == 236


def test_harm_only_claim_has_180_tasks() -> None:
    tasks = list(_claim(include_needs_caution=False))
    assert len(tasks) == 180
    assert UNSAFE_COUNT_HARM_ONLY == 180


def test_each_task_is_an_aegis_task() -> None:
    assert all(isinstance(t, AegisTask) for t in _claim(limit=5))


def test_returns_a_security_claim() -> None:
    assert isinstance(_claim(limit=1), SecurityClaim)


def test_limit_caps_task_count() -> None:
    assert len(list(_claim(limit=20))) == 20


def test_negative_limit_raises() -> None:
    with pytest.raises(ValueError, match="limit must be >= 0"):
        _claim(limit=-1)


def test_category_filter_keeps_only_matching_rows() -> None:
    tasks = list(_claim(categories=["Violence"]))
    assert tasks  # non-empty
    for t in tasks:
        assert "Violence" in t.risk_category


def test_category_filter_with_unknown_category_raises() -> None:
    with pytest.raises(ValueError, match="No tasks produced"):
        _claim(categories=["No Such Category"])


def test_requires_judge_or_judge_llm_config() -> None:
    with pytest.raises(ValueError, match="`judge` or `judge_llm_config`"):
        aegis_claim()


def test_accepts_judge_llm_config_alone() -> None:
    cfg = LLMConfig(model="openai/gpt-4o", api_base="http://example.invalid", api_key="sk-x")
    claim = aegis_claim(judge_llm_config=cfg, limit=3)
    # Tasks are created (the AegisSafetyJudge is built lazily; no call happens).
    assert len(list(claim)) == 3


def test_claim_is_re_iterable() -> None:
    claim = _claim(limit=10)
    assert len(list(claim)) == len(list(claim)) == 10


def test_pinned_provenance_constants() -> None:
    assert DATASET_REPO_ID == "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"
    assert DATASET_REVISION == "bd96d862068e47630197de64eb91f8d1481ff3e0"
    assert DATASET_SPLIT == "test"
