"""Drive the optimizer's event state machine end to end, offline.

These tests exercise the inversion: RunStart selects a node, PreCall delivers each
sub-task turn, PostCall/observable carries the reply, RunEnd scores the node and
grows the tree. The scripted LLM makes the vendored decomposition/judge/feedback
helpers deterministic.
"""

from __future__ import annotations

import pytest
from conftest import (
    ScriptedLLM,
    dispatch,
    goal,
    make_controllable,
    post_call,
    pre_call,
    response_observable,
    run_end,
    run_end_no_feedback,
    run_start,
)
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    RunEndResponse,
)

from attack_anything_optimizer import AttackAnythingConfig, AttackAnythingOptimizer


async def _init(opt: AttackAnythingOptimizer, llm: ScriptedLLM, ctrls=None) -> None:
    ctrls = ctrls if ctrls is not None else [make_controllable()]
    await opt.initialize(goal(), ctrls, [], llm)  # type: ignore[arg-type]


async def _drive_conversation(opt, ctrl, traj, reply: str, max_turns: int = 40):
    """Answer PreCalls until the optimizer declines (its loop's `break`).

    Returns the list of injected values (one per delivered turn).
    """
    injected: list[str] = []
    for _ in range(max_turns):
        resp = await dispatch(opt, pre_call(ctrl))
        if isinstance(resp, ControllableNoInjection):
            break
        assert isinstance(resp, ControllableInjection)
        injected.append(resp.value)
        # The framework records the reply on the trajectory; mirror that.
        traj.push(response_observable(reply))
        await dispatch(opt, post_call(ctrl, request=resp.value, answer=reply))
    return injected


@pytest.mark.asyncio
async def test_full_decomposition_cycle_injects_each_subtask() -> None:
    """A run walks all four sub-tasks in one conversation, one turn each when the
    victim complies, and the run ends by declining."""
    llm = ScriptedLLM(
        subtasks=["s1", "s2", "s3", "s4"],
        judge_labels=["affirmative"] * 8,
    )
    opt = AttackAnythingOptimizer(config=AttackAnythingConfig(n_iterations=3))
    ctrl = make_controllable()
    await _init(opt, llm)

    from conftest import FakeTrajectory

    traj = FakeTrajectory()
    await dispatch(opt, run_start(traj))
    injected = await _drive_conversation(opt, ctrl, traj, reply="Sure, here is the detail.")
    # Four sub-tasks, each answered affirmatively on the first turn -> 4 injections.
    assert len(injected) == 4
    # Each sub-task's wrapped text appears in its injected message.
    assert any("s1" in m for m in injected)
    assert any("s4" in m for m in injected)

    end = await dispatch(opt, run_end(success=False, score=0.0))
    assert isinstance(end, RunEndResponse)


@pytest.mark.asyncio
async def test_framework_success_is_authoritative_and_stops() -> None:
    """A framework success (n_early_stop_successes=1) ends the task with done=True,
    regardless of the internal judge."""
    llm = ScriptedLLM(judge_labels=["refusal"] * 20)  # internal judge says no...
    opt = AttackAnythingOptimizer(
        config=AttackAnythingConfig(n_early_stop_successes=1, n_iterations=3)
    )
    ctrl = make_controllable()
    await _init(opt, llm)

    from conftest import FakeTrajectory

    traj = FakeTrajectory()
    await dispatch(opt, run_start(traj))
    await _drive_conversation(opt, ctrl, traj, reply="I cannot help with that.")
    end = await dispatch(opt, run_end(success=True, score=1.0))  # ...but the Task says yes
    assert isinstance(end, RunEndResponse)
    assert end.done is True
    assert opt._succeeded is True


@pytest.mark.asyncio
async def test_no_feedback_run_never_self_certifies() -> None:
    """With evaluation=None (include_feedback=False), the optimizer cannot mark
    itself succeeded off its own judge."""
    llm = ScriptedLLM(judge_labels=["affirmative"] * 20)
    opt = AttackAnythingOptimizer(config=AttackAnythingConfig(n_early_stop_successes=1))
    ctrl = make_controllable()
    await _init(opt, llm)

    from conftest import FakeTrajectory

    traj = FakeTrajectory()
    await dispatch(opt, run_start(traj))
    await _drive_conversation(opt, ctrl, traj, reply="Sure, here it is.")
    await dispatch(opt, run_end_no_feedback())
    assert opt._succeeded is False


@pytest.mark.asyncio
async def test_response_channel_is_declined() -> None:
    """A `response` controllable is always declined (never rewrite the victim)."""
    llm = ScriptedLLM()
    opt = AttackAnythingOptimizer()
    await _init(opt, llm)
    from conftest import FakeTrajectory

    traj = FakeTrajectory()
    await dispatch(opt, run_start(traj))
    resp = await dispatch(opt, pre_call(make_controllable(name="response")))
    assert isinstance(resp, ControllableNoInjection)


@pytest.mark.asyncio
async def test_json_channel_is_declined() -> None:
    """A non-free-text (json) primary channel is declined -- the attack is text."""
    llm = ScriptedLLM()
    opt = AttackAnythingOptimizer()
    ctrl = make_controllable(name="skill", value_type="json")
    await _init(opt, llm, ctrls=[ctrl])
    from conftest import FakeTrajectory

    traj = FakeTrajectory()
    await dispatch(opt, run_start(traj))
    resp = await dispatch(opt, pre_call(ctrl))
    assert isinstance(resp, ControllableNoInjection)


@pytest.mark.asyncio
async def test_never_injects_empty_string() -> None:
    """No injected value is ever empty (the chatbot target kills a run after 3)."""
    llm = ScriptedLLM(subtasks=["only step"], judge_labels=["affirmative"])
    opt = AttackAnythingOptimizer(config=AttackAnythingConfig(n_iterations=2))
    ctrl = make_controllable()
    await _init(opt, llm)
    from conftest import FakeTrajectory

    traj = FakeTrajectory()
    await dispatch(opt, run_start(traj))
    injected = await _drive_conversation(opt, ctrl, traj, reply="Sure.")
    assert injected
    assert all(v.strip() for v in injected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "toggles",
    [
        {"use_decomposition": False},
        {"use_feedback": False},
        {"use_tree_search": False},
        {"use_archive": False},
        {"use_decomposition": False, "use_feedback": False},
    ],
)
async def test_toggle_matrix_runs_without_error(toggles: dict) -> None:
    """Each ablation config drives a full run to completion (the ablation ladder)."""
    llm = ScriptedLLM(judge_labels=["refusal", "unclear", "affirmative"] * 10)
    opt = AttackAnythingOptimizer(config=AttackAnythingConfig(n_iterations=2, **toggles))
    ctrl = make_controllable()
    await _init(opt, llm)
    from conftest import FakeTrajectory

    traj = FakeTrajectory()
    await dispatch(opt, run_start(traj))
    await _drive_conversation(opt, ctrl, traj, reply="Here is a partial answer.")
    end = await dispatch(opt, run_end(success=False, score=0.3))
    assert isinstance(end, RunEndResponse)


@pytest.mark.asyncio
async def test_no_llm_degrades_to_rule_based() -> None:
    """A noop LLM client (BudgetExhausted, cost 0) degrades to the vendored
    rule-based paths instead of crashing."""
    from superred.core.llm import LLMClient

    opt = AttackAnythingOptimizer(config=AttackAnythingConfig(n_iterations=2))
    ctrl = make_controllable()
    await opt.initialize(goal(), [ctrl], [], LLMClient._make_noop())
    from conftest import FakeTrajectory

    traj = FakeTrajectory()
    await dispatch(opt, run_start(traj))
    # Seeding used rule-based decomposition; a node is available and injects.
    first = await dispatch(opt, pre_call(ctrl))
    assert isinstance(first, (ControllableInjection, ControllableNoInjection))
    assert opt._llm_available is False
