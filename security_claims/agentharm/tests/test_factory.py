"""Factory: claim sizes, category filters, named factories, target factory."""

from __future__ import annotations

import pytest
from inspect_agent_target.target import InspectAgentTarget
from superred.core.controller import TargetFactory

from security_claim_agentharm.factory import (
    agentharm_category_claim,
    agentharm_claim,
    agentharm_fraud_claim,
    agentharm_hate_claim,
    agentharm_target_factory,
    agentharm_total_claim,
)

_M = "openai/gpt-4o-mini"


def _count(claim) -> int:  # noqa: ANN001
    return len(list(claim))


def test_total_claim_is_176() -> None:
    assert _count(agentharm_total_claim(target_model=_M)) == 176


def test_named_category_counts() -> None:
    assert _count(agentharm_fraud_claim(target_model=_M)) == 24
    assert _count(agentharm_hate_claim(target_model=_M)) == 20


def test_claim_category_filter() -> None:
    fraud = list(agentharm_claim(target_model=_M, categories=["Fraud"]))
    assert len(fraud) == 24
    assert all(t.category == "Fraud" for t in fraud)


def test_behavior_ids_filter() -> None:
    assert _count(agentharm_claim(target_model=_M, behavior_ids=["1-1"])) == 1


def test_unknown_category_raises() -> None:
    with pytest.raises(ValueError, match="Unknown AgentHarm category"):
        agentharm_claim(target_model=_M, categories=["Nope"])
    with pytest.raises(ValueError, match="Unknown AgentHarm category"):
        agentharm_category_claim("Nope", target_model=_M)


def test_target_factory_builds_general_target() -> None:
    tf = agentharm_target_factory(target_model=_M, api_base="b", api_key="k", concurrency=3)
    assert isinstance(tf, TargetFactory)
    assert tf.concurrency == 3
    assert isinstance(tf.create(), InspectAgentTarget)
