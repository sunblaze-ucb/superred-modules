"""Budget propagation and clean attempt-exhaustion stop for :class:`MuzzleOptimizer`.

Every ``self.llm`` call the optimizer makes (the brain helpers and the offline PAIR loop) may
raise :class:`BudgetExhaustedError` once the controller-enforced spend cap is hit. The optimizer
must let that exception propagate -- it is the controller's job, not the optimizer's, to catch it
and record the task as ``budget_exhausted``. Independently, when no budget error occurs the
optimizer must come to rest on its own ``max_attempts`` cap with ``done=True``.

The controller's ``max_runs_per_task`` is the hard backstop that also bounds OBSERVE + PROBE
runs (which ``max_attempts``, counting only scored ATTACK runs, does not); these tests exercise
the optimizer's own two stop conditions, which sit inside that backstop.
"""

from __future__ import annotations

import pytest

from muzzle_optimizer.optimizer import MuzzleOptimizer
from superred.core.types.events import RunEndResponse
from superred.core.types.goal import Goal
from superred.core.types.llm import BudgetExhaustedError, LLMUsage

from conftest import (
    TOOL_A_TAG,
    FakeTrajectory,
    ScriptedLLM,
    dispatch_event,
    make_controllable,
    post_call,
    run_end,
    run_start,
)


def _budget_exhausted(_messages: list[dict[str, str]]) -> str:
    """A ScriptedLLM role override that simulates the controller cutting the optimizer off."""
    raise BudgetExhaustedError("LLM budget exhausted", usage=LLMUsage())


async def _run(optimizer: MuzzleOptimizer, ctrl: object, end_event: object) -> object:
    """Drive one full run over a content post-call surface, recording any injection."""
    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))
    event = post_call(ctrl)  # type: ignore[arg-type]
    traj.push(event)
    traj.push(await dispatch_event(optimizer, event))
    return await dispatch_event(optimizer, end_event)


@pytest.mark.asyncio
async def test_budget_exhausted_in_summarizer_propagates() -> None:
    """A BudgetExhaustedError from the OBSERVE summarizer is not swallowed by _dispatch."""
    ctrl = make_controllable("tool:web", TOOL_A_TAG)
    llm = ScriptedLLM(overrides={"summarizer": _budget_exhausted})
    optimizer = MuzzleOptimizer(pair_judge_method="gcg", pair_n_streams=1, pair_n_iterations=1)
    await optimizer.initialize(Goal(description="g"), [ctrl], [], llm)

    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))
    # OBSERVE's run-end triggers the summarizer; the error must escape dispatch_event unchanged
    # so the controller (the backstop) records budget_exhausted.
    with pytest.raises(BudgetExhaustedError):
        await dispatch_event(optimizer, run_end())


@pytest.mark.asyncio
async def test_budget_exhausted_in_pair_attacker_propagates() -> None:
    """A BudgetExhaustedError from the offline PAIR attacker propagates at ATTACK run-start."""
    ctrl = make_controllable("tool:web", TOOL_A_TAG)
    llm = ScriptedLLM(overrides={"pair_attacker": _budget_exhausted})
    optimizer = MuzzleOptimizer(pair_judge_method="gcg", pair_n_streams=1, pair_n_iterations=1)
    await optimizer.initialize(Goal(description="g"), [ctrl], [], llm)

    await _run(optimizer, ctrl, run_end())  # OBSERVE -> PROBE (default brain roles)
    await _run(optimizer, ctrl, run_end())  # PROBE -> ATTACK (marker surfaces)
    assert optimizer._phase == "ATTACK"

    # The next run-start crafts the payload offline via PAIR; the attacker call raises and the
    # error must propagate out of dispatch_event rather than being caught.
    traj = FakeTrajectory()
    with pytest.raises(BudgetExhaustedError):
        await dispatch_event(optimizer, run_start(traj))


@pytest.mark.asyncio
async def test_reaching_max_attempts_stops_done_without_success() -> None:
    """With max_attempts=1 the first unsuccessful ATTACK run stops cleanly via exhaustion."""
    ctrl = make_controllable("tool:web", TOOL_A_TAG)
    llm = ScriptedLLM()
    optimizer = MuzzleOptimizer(
        max_attempts=1, pair_judge_method="gcg", pair_n_streams=1, pair_n_iterations=1
    )
    await optimizer.initialize(Goal(description="g"), [ctrl], [], llm)

    await _run(optimizer, ctrl, run_end())  # OBSERVE -> PROBE
    await _run(optimizer, ctrl, run_end())  # PROBE -> ATTACK
    assert optimizer._phase == "ATTACK"

    # A non-successful scored ATTACK run hits the single-attempt cap and stops.
    end = await _run(optimizer, ctrl, run_end(success=False))
    assert isinstance(end, RunEndResponse)
    assert end.done is True
    assert optimizer._attempt_index == optimizer._max_attempts == 1
    # The stop is attempt exhaustion, never a self-certified success.
    assert optimizer._succeeded is False
    assert optimizer._is_done() is True
