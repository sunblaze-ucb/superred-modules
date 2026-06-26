"""Factory enumeration, filters, roll-ups, and target-factory wiring."""

from __future__ import annotations

import sys
import types

import pytest
from conftest import dataset_root, requires_dataset
from superred.core.controller import TargetFactory

from security_claim_dtap.factory import (
    dtap_benign_claim,
    dtap_claim,
    dtap_claudecode_target_factory,
    dtap_combined_claim,
    dtap_direct_claim,
    dtap_domain_claim,
    dtap_indirect_claim,
    dtap_openclaw_target_factory,
    dtap_risk_claim,
)
from security_claim_dtap.task import DtapTask

# ---------------------------------------------------------------------------
# Enumeration + filters (dataset-dependent)
# ---------------------------------------------------------------------------


@requires_dataset
def test_dtap_claim_travel_nonempty() -> None:
    claim = dtap_claim(domains=["travel"], dataset_root=str(dataset_root()))
    tasks = list(claim)
    assert len(tasks) > 0
    assert all(isinstance(t, DtapTask) for t in tasks)


@requires_dataset
def test_domain_claim_matches_dtap_claim() -> None:
    a = list(dtap_domain_claim("travel", dataset_root=str(dataset_root())))
    b = list(dtap_claim(domains=["travel"], dataset_root=str(dataset_root())))
    assert len(a) == len(b) > 0


@requires_dataset
def test_direct_claim_all_malicious_direct() -> None:
    tasks = list(dtap_direct_claim(domains=["travel"], dataset_root=str(dataset_root())))
    assert len(tasks) > 0
    assert all(t.is_malicious and t.threat_model == "direct" for t in tasks)


@requires_dataset
def test_indirect_claim_all_malicious_indirect() -> None:
    tasks = list(dtap_indirect_claim(domains=["travel"], dataset_root=str(dataset_root())))
    assert len(tasks) > 0
    assert all(t.is_malicious and t.threat_model == "indirect" for t in tasks)


@requires_dataset
def test_benign_claim_all_benign() -> None:
    tasks = list(dtap_benign_claim(domains=["travel"], dataset_root=str(dataset_root())))
    assert len(tasks) > 0
    assert all(not t.is_malicious for t in tasks)


@requires_dataset
def test_risk_claim_filters_risk_category() -> None:
    tasks = list(
        dtap_risk_claim("booking-abuse", domains=["travel"], dataset_root=str(dataset_root()))
    )
    assert len(tasks) > 0
    assert all(t.risk_category == "booking-abuse" for t in tasks)


@requires_dataset
def test_types_partition_sums_to_total() -> None:
    root = str(dataset_root())
    total = len(list(dtap_claim(domains=["travel"], dataset_root=root)))
    mal = len(list(dtap_claim(domains=["travel"], types=["malicious"], dataset_root=root)))
    ben = len(list(dtap_claim(domains=["travel"], types=["benign"], dataset_root=root)))
    assert mal + ben == total
    assert mal > 0 and ben > 0


@requires_dataset
def test_combined_claim_chains() -> None:
    root = str(dataset_root())
    direct = dtap_direct_claim(domains=["travel"], dataset_root=root)
    indirect = dtap_indirect_claim(domains=["travel"], dataset_root=root)
    combined = dtap_combined_claim([direct, indirect])
    assert len(list(combined)) == len(list(direct)) + len(list(indirect))


@requires_dataset
def test_judge_creds_threaded_into_tasks() -> None:
    claim = dtap_claim(
        domains=["travel"],
        types=["malicious"],
        dataset_root=str(dataset_root()),
        judge_model="m",
        judge_api_base="b",
        judge_api_key="k",
    )
    task = next(iter(claim))
    assert isinstance(task, DtapTask)
    assert task._judge_model == "m"
    assert task._judge_api_base == "b"
    assert task._judge_api_key == "k"


@requires_dataset
def test_empty_filter_raises_value_error() -> None:
    with pytest.raises(ValueError, match="no tasks"):
        dtap_claim(
            domains=["travel"],
            risk_categories=["this-risk-does-not-exist"],
            dataset_root=str(dataset_root()),
        )


# ---------------------------------------------------------------------------
# Target factories (no dataset, no Docker)
# ---------------------------------------------------------------------------


def test_claudecode_target_factory_shape() -> None:
    tf = dtap_claudecode_target_factory(model="m", api_base="b", api_key="k")
    assert isinstance(tf, TargetFactory)
    assert tf.concurrency == 1
    assert callable(tf.create)


def test_openclaw_target_factory_concurrency() -> None:
    tf = dtap_openclaw_target_factory(model="m", api_base="b", api_key="k", concurrency=4)
    assert isinstance(tf, TargetFactory)
    assert tf.concurrency == 4


def test_claudecode_factory_create_lazy_imports_and_wires(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTarget:
        def __init__(self, *, model, api_base, api_key, state_root) -> None:
            captured.update(model=model, api_base=api_base, api_key=api_key, state_root=state_root)

    mod = types.ModuleType("dtap_claudecode_target")
    mod.ClaudeCodeDtapTarget = FakeTarget  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dtap_claudecode_target", mod)

    tf = dtap_claudecode_target_factory(
        model="gpt", api_base="http://p", api_key="sk", state_root="/state"
    )
    inst = tf.create()
    assert isinstance(inst, FakeTarget)
    assert captured == {
        "model": "gpt",
        "api_base": "http://p",
        "api_key": "sk",
        "state_root": "/state",
    }


def test_openclaw_factory_create_lazy_imports_and_wires(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTarget:
        def __init__(self, *, model, api_base, api_key, state_root) -> None:
            captured.update(model=model, api_base=api_base, api_key=api_key, state_root=state_root)

    mod = types.ModuleType("dtap_openclaw_target")
    mod.OpenClawDtapTarget = FakeTarget  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dtap_openclaw_target", mod)

    tf = dtap_openclaw_target_factory(model="gpt", api_base=None, api_key=None)
    inst = tf.create()
    assert isinstance(inst, FakeTarget)
    assert captured == {"model": "gpt", "api_base": None, "api_key": None, "state_root": None}
