"""Drive the pump (optimizer) across a simulated controller run-loop, offline.

The pump maps planner Units onto superred runs (fresh unit -> new run, non-fresh
-> continuation). ``run_task`` simulates the controller: for each run it opens a
fresh trajectory, answers PreCalls until the optimizer declines, feeds replies
back on the trajectory, and delivers a RunEnd verdict.
"""

from __future__ import annotations

import pytest
from conftest import (
    FakeTrajectory,
    dispatch,
    goal,
    make_controllable,
    post_call,
    pre_call,
    response_observable,
    run_end,
    run_start,
)
from superred.core.llm import LLMClient
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    RunEndResponse,
)

from attack_anything_optimizer import AttackAnythingConfig, AttackAnythingOptimizer


async def run_task(opt, ctrl, reply_fn, success_fn=None, max_runs=300, max_turns=60):
    """Simulate the controller loop. Returns a list of (injected_values, done)."""
    runs: list[tuple[list[str], bool]] = []
    for run_idx in range(max_runs):
        traj = FakeTrajectory()
        await dispatch(opt, run_start(traj))
        injected: list[str] = []
        for _ in range(max_turns):
            resp = await dispatch(opt, pre_call(ctrl))
            if isinstance(resp, ControllableNoInjection):
                break
            assert isinstance(resp, ControllableInjection)
            injected.append(resp.value)
            reply = reply_fn(resp.value, run_idx, len(injected))
            traj.push(response_observable(reply))
            await dispatch(opt, post_call(ctrl, request=resp.value, answer=reply))
        success = bool(success_fn(injected, run_idx)) if success_fn else False
        end = await dispatch(opt, run_end(success=success))
        assert isinstance(end, RunEndResponse)
        runs.append((injected, end.done))
        if end.done:
            break
    return runs


async def _init(opt, ctrls=None):
    ctrls = ctrls if ctrls is not None else [make_controllable()]
    # Noop client -> the planner uses vendored rule-based paths (no network).
    await opt.initialize(goal(), ctrls, [], LLMClient._make_noop())


@pytest.mark.asyncio
async def test_default_run_completes_and_injects_subtasks() -> None:
    opt = AttackAnythingOptimizer(
        config=AttackAnythingConfig(n_iterations=2, n_steps=3, max_turns=2)
    )
    ctrl = make_controllable()
    await _init(opt)
    runs = await run_task(opt, ctrl, lambda v, r, t: "I'm sorry, I can't help with that.")
    # It ran several runs, injected non-empty messages, and terminated cleanly.
    assert runs
    assert any(inj for inj, _ in runs)
    assert all(v.strip() for inj, _ in runs for v in inj)


@pytest.mark.asyncio
async def test_framework_success_on_assembly_is_authoritative() -> None:
    """When the victim complies, the assembly run's framework success ends the task."""
    opt = AttackAnythingOptimizer(
        config=AttackAnythingConfig(n_iterations=2, n_steps=2, max_turns=1)
    )
    ctrl = make_controllable()
    await _init(opt)
    # Victim complies on every message; the LAST run (assembly) is judged success.
    runs = await run_task(
        opt,
        ctrl,
        reply_fn=lambda v, r, t: "Sure, here are the detailed steps.",
        success_fn=lambda inj, r: True,  # framework judges each run success
    )
    assert runs[-1][1] is True  # done


@pytest.mark.asyncio
async def test_no_feedback_run_never_self_certifies() -> None:
    """With framework success always False, the task never reports done via its own judge."""
    opt = AttackAnythingOptimizer(
        config=AttackAnythingConfig(n_iterations=1, n_steps=2, max_turns=1)
    )
    ctrl = make_controllable()
    await _init(opt)
    runs = await run_task(
        opt,
        ctrl,
        reply_fn=lambda v, r, t: "Sure, here it is.",
        success_fn=lambda inj, r: False,
        max_runs=60,
    )
    # The optimizer never claimed framework success off its own internal judge...
    assert opt._framework_succeeded is False
    # ...so any `done` came only from the planner exhausting its search, which
    # can happen at most on the final run (never mid-search self-certification).
    done_flags = [done for _, done in runs]
    assert done_flags.count(True) <= 1
    if True in done_flags:
        assert done_flags[-1] is True


@pytest.mark.asyncio
async def test_response_and_json_channels_declined() -> None:
    opt = AttackAnythingOptimizer()
    await _init(opt, ctrls=[make_controllable(name="response")])
    traj = FakeTrajectory()
    await dispatch(opt, run_start(traj))
    resp = await dispatch(opt, pre_call(make_controllable(name="response")))
    assert isinstance(resp, ControllableNoInjection)

    opt2 = AttackAnythingOptimizer()
    json_ctrl = make_controllable(name="skill", value_type="json")
    await _init(opt2, ctrls=[json_ctrl])
    traj2 = FakeTrajectory()
    await dispatch(opt2, run_start(traj2))
    resp2 = await dispatch(opt2, pre_call(json_ctrl))
    assert isinstance(resp2, ControllableNoInjection)


@pytest.mark.asyncio
async def test_never_injects_empty_string() -> None:
    opt = AttackAnythingOptimizer(
        config=AttackAnythingConfig(n_iterations=1, n_steps=2, max_turns=2)
    )
    ctrl = make_controllable()
    await _init(opt)
    runs = await run_task(opt, ctrl, lambda v, r, t: "Sure.")
    assert all(v.strip() for inj, _ in runs for v in inj)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cfg",
    [
        AttackAnythingConfig(n_iterations=1, n_steps=2, max_turns=1, use_decomposition=False),
        AttackAnythingConfig(n_iterations=1, n_steps=2, max_turns=1, use_feedback=False),
        AttackAnythingConfig(n_iterations=1, n_steps=2, max_turns=1, use_tree_search=False),
        AttackAnythingConfig(n_iterations=1, n_steps=2, max_turns=1, use_archive=False),
        AttackAnythingConfig(
            n_steps=2,
            recursive_leaf_attack=True,
            recursive_max_depth=1,
            recursive_branch=2,
            recursive_wrappers_per_leaf=2,
            max_target_queries_per_goal=12,
            n_iterations=1,
        ),
        AttackAnythingConfig(
            goal_as_root=True,
            fallback_enabled=True,
            recursive_max_depth=1,
            recursive_branch=2,
            recursive_wrappers_per_leaf=2,
            max_target_queries_per_goal=15,
        ),
        AttackAnythingConfig(n_iterations=1, n_steps=2, max_turns=2, turn_independent=True),
    ],
)
async def test_mode_matrix_runs_without_error(cfg: AttackAnythingConfig) -> None:
    opt = AttackAnythingOptimizer(config=cfg)
    ctrl = make_controllable()
    await _init(opt)
    runs = await run_task(opt, ctrl, lambda v, r, t: "I refuse to help with that.", max_runs=200)
    assert runs  # completed without raising
