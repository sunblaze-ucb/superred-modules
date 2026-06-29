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


def test_total_claim_default_excludes_unwinnable_bases() -> None:
    # benign_cover=True (default) drops bases 22 + 49 (structurally unwinnable), 8 behaviours.
    assert _count(agentharm_total_claim(target_model=_M)) == 168


def test_total_claim_direct_mode_is_176() -> None:
    assert _count(agentharm_total_claim(target_model=_M, benign_cover=False)) == 176


def test_named_category_counts() -> None:
    assert _count(agentharm_fraud_claim(target_model=_M)) == 24
    assert _count(agentharm_hate_claim(target_model=_M)) == 20


def test_claim_category_filter() -> None:
    fraud = list(agentharm_claim(target_model=_M, categories=["Fraud"]))
    assert len(fraud) == 24
    assert all(t.category == "Fraud" for t in fraud)


def test_behavior_ids_filter() -> None:
    assert _count(agentharm_claim(target_model=_M, behavior_ids=["1-1"])) == 1


class _FakeTarget:
    def __init__(self) -> None:
        self.cfg: dict[str, str] = {}

    def set_config(self, name: str, value: str) -> None:
        self.cfg[name] = value


@pytest.mark.asyncio
async def test_benign_cover_is_the_default() -> None:
    [t] = list(agentharm_claim(target_model=_M, behavior_ids=["1-1"]))
    assert t.is_benign_cover is True
    # Goal stays the harmful behaviour; the delivered user task is the benign cover.
    tgt = _FakeTarget()
    await t.configure_target(tgt)
    assert tgt.cfg["user_prompt"].strip() != t.goal.description.strip()


@pytest.mark.asyncio
async def test_benign_cover_false_restores_direct_request() -> None:
    [t] = list(agentharm_claim(target_model=_M, behavior_ids=["1-1"], benign_cover=False))
    assert t.is_benign_cover is False
    tgt = _FakeTarget()
    await t.configure_target(tgt)
    assert tgt.cfg["user_prompt"].strip() == t.goal.description.strip()


def test_counts_unchanged_in_both_cover_modes() -> None:
    assert _count(agentharm_claim(target_model=_M, categories=["Fraud"], benign_cover=True)) == 24
    assert _count(agentharm_claim(target_model=_M, categories=["Fraud"], benign_cover=False)) == 24


def test_explicit_excluded_behavior_id_raises_clear_error() -> None:
    # Asking for an excluded (unwinnable) behaviour by id in benign-cover mode is a clear
    # error, not a silent empty claim; benign_cover=False serves it as a direct request.
    with pytest.raises(ValueError, match="excluded from benign-cover"):
        agentharm_claim(target_model=_M, behavior_ids=["49-1"])
    assert _count(agentharm_claim(target_model=_M, behavior_ids=["49-1"], benign_cover=False)) == 1


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
