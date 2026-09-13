"""Factory tests: filtering, judge requirement, roll-ups, target factory."""

from __future__ import annotations

import pytest
from safeclawbench_claim.factory import (
    safeclawbench_claim,
    safeclawbench_combined_claim,
    safeclawbench_family_claim,
    safeclawbench_target_factory,
)
from safeclawbench_claim.judge import StubCoreFailJudge
from superred.core.controller import TargetFactory

_JUDGE = StubCoreFailJudge()


def _tasks(claim):
    return list(claim)


def test_requires_a_judge():
    with pytest.raises(ValueError, match="requires a CoreFail judge"):
        safeclawbench_claim()


def test_full_claim_has_600_tasks():
    claim = safeclawbench_claim(judge=_JUDGE)
    assert len(_tasks(claim)) == 600


def test_family_filter():
    claim = safeclawbench_claim(judge=_JUDGE, attack_types=["DPI"])
    tasks = _tasks(claim)
    assert len(tasks) == 100
    assert all(t.case.attack_type == "DPI" for t in tasks)


def test_unknown_family_rejected():
    with pytest.raises(ValueError, match="Unknown attack_types"):
        safeclawbench_claim(judge=_JUDGE, attack_types=["NOPE"])


def test_limit_and_difficulty_filter():
    claim = safeclawbench_claim(judge=_JUDGE, difficulties=["hard"], limit=5)
    assert len(_tasks(claim)) == 5


def test_task_id_filter():
    all_tasks = _tasks(safeclawbench_claim(judge=_JUDGE, attack_types=["MEX"]))
    some_id = all_tasks[0].case.task_id
    claim = safeclawbench_claim(judge=_JUDGE, task_ids=[some_id])
    tasks = _tasks(claim)
    assert len(tasks) == 1
    assert tasks[0].case.task_id == some_id


def test_exclude_seed_drops_seed_cases():
    with_seed = _tasks(safeclawbench_claim(judge=_JUDGE))
    without_seed = _tasks(safeclawbench_claim(judge=_JUDGE, exclude_seed=True))
    assert len(without_seed) < len(with_seed)


def test_empty_filter_raises():
    with pytest.raises(ValueError, match="no tasks"):
        safeclawbench_claim(judge=_JUDGE, attack_types=["DPI"], harm_types=["nonexistent"])


def test_family_claim_roll_up():
    claim = safeclawbench_family_claim("IPI", judge=_JUDGE)
    assert all(t.case.attack_type == "IPI" for t in _tasks(claim))


def test_combined_claim():
    dpi = safeclawbench_family_claim("DPI", judge=_JUDGE)
    ipi = safeclawbench_family_claim("IPI", judge=_JUDGE)
    combined = safeclawbench_combined_claim([dpi, ipi])
    assert len(_tasks(combined)) == 200


def test_target_factory_returns_target_factory():
    factory = safeclawbench_target_factory()
    assert isinstance(factory, TargetFactory)
