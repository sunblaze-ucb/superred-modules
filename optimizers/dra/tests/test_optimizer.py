"""Tests for DraOptimizer's SuperRed integration."""

from __future__ import annotations

from random import Random

import pytest

from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal

from dra_optimizer.builder import OFFICIAL_SYSTEM_PROMPT
from dra_optimizer.optimizer import DraOptimizer
from tests.conftest import (
    FakeReadableTrajectory,
    MODEL_TAG,
    SYSTEM_PROMPT_TAG,
    USER_TAG,
    fake_llm,
    make_controllable,
    make_observable,
    model_observable,
)


def success_eval() -> EvaluationResult:
    return EvaluationResult(
        success=True,
        primary_score=Score(value=1.0, name="primary"),
        sub_scores={},
        rationale="task succeeded",
    )


def failure_eval() -> EvaluationResult:
    return EvaluationResult(
        success=False,
        primary_score=Score(value=1.0, name="primary"),
        sub_scores={},
        rationale="task failed",
    )


async def init_optimizer(
    *,
    controllables: list | None = None,
    observables: list | None = None,
    max_attempts: int = 3,
    max_no_signal_runs: int = 0,
    model: str = "auto",
) -> DraOptimizer:
    opt = DraOptimizer(
        max_attempts=max_attempts,
        max_no_signal_runs=max_no_signal_runs,
        model=model,
        rng=Random(9),
    )
    await opt.initialize(
        goal=Goal(description="alpha beta"),
        controllables=controllables if controllables is not None else [make_controllable()],
        observables=observables if observables is not None else [],
        llm_client=fake_llm(),
    )
    return opt


@pytest.mark.asyncio
async def test_default_model_is_llama_without_model_observable() -> None:
    opt = await init_optimizer()
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    assert opt._model == "llama"
    assert opt._current_attack is not None
    assert opt._current_attack.model == "llama"


@pytest.mark.asyncio
async def test_model_observable_selects_official_suffix_family() -> None:
    opt = await init_optimizer(observables=[model_observable("gpt-4o-2024-05-13")])
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    assert opt._model == "gpt-4o"
    assert opt._current_attack is not None
    assert opt._current_attack.model == "gpt-4o"


@pytest.mark.asyncio
async def test_chatbot_target_shape_injects_system_prompt_then_user_prompt() -> None:
    opt = await init_optimizer(
        controllables=[
            make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            make_controllable("user_message", USER_TAG),
        ],
    )
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    sp = await opt.on_event(
        ControllablePreCallEvent(
            controllable=make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            request="default",
        )
    )
    user = await opt.on_event(
        ControllablePreCallEvent(
            controllable=make_controllable("user_message", USER_TAG),
            request="user message",
        )
    )

    assert isinstance(sp, ControllableInjection)
    assert sp.value == OFFICIAL_SYSTEM_PROMPT
    assert isinstance(user, ControllableInjection)
    assert OFFICIAL_SYSTEM_PROMPT not in user.value
    assert "construct a sentence named Q1" in user.value


@pytest.mark.asyncio
async def test_system_prompt_precall_does_not_steal_user_lock_when_not_writable() -> None:
    opt = await init_optimizer(controllables=[make_controllable("user_message", USER_TAG)])
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    sp = await opt.on_event(
        ControllablePreCallEvent(
            controllable=make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            request="default",
        )
    )
    user = await opt.on_event(
        ControllablePreCallEvent(
            controllable=make_controllable("user_message", USER_TAG),
            request="user message",
        )
    )

    assert isinstance(sp, ControllableNoInjection)
    assert isinstance(user, ControllableInjection)


@pytest.mark.asyncio
async def test_response_precall_is_skipped_without_stealing_user_lock() -> None:
    opt = await init_optimizer()
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    response_pre = await opt.on_event(
        ControllablePreCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request="assistant answer",
        )
    )
    user = await opt.on_event(
        ControllablePreCallEvent(
            controllable=make_controllable("user_message", USER_TAG),
            request="user message",
        )
    )

    assert isinstance(response_pre, ControllableNoInjection)
    assert isinstance(user, ControllableInjection)


@pytest.mark.asyncio
async def test_trajectory_response_is_used_before_postcall_fallback() -> None:
    traj = FakeReadableTrajectory()
    opt = await init_optimizer()
    await opt.on_event(RunStartEvent(trajectory=traj))
    pre = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )
    assert isinstance(pre, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=pre.value,
            answer="I'm sorry, I cannot help with alpha beta",
        )
    )
    traj.push(
        ObservableEvent(
            observable=make_observable("response"),
            content="alpha beta detailed answer",
        )
    )

    end = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert isinstance(end, RunEndResponse)
    assert end.done is True
    assert opt._succeeded is True
    assert opt._last_check is not None
    assert opt._last_check.jailbroken_gcg is True


@pytest.mark.asyncio
async def test_postcall_answer_is_used_when_trajectory_response_is_not_visible() -> None:
    opt = await init_optimizer()
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    pre = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )
    assert isinstance(pre, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=pre.value,
            answer="alpha beta detailed answer",
        )
    )

    end = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is True
    assert opt._succeeded is True


@pytest.mark.asyncio
async def test_non_success_framework_evaluation_blocks_local_success() -> None:
    traj = FakeReadableTrajectory()
    traj.push(ObservableEvent(observable=make_observable("response"), content="alpha beta answer"))
    opt = await init_optimizer(max_attempts=2)
    await opt.on_event(RunStartEvent(trajectory=traj))

    end = await opt.on_event(RunEndEvent(evaluation=failure_eval(), security_domain=USER_TAG))

    assert end.done is False
    assert opt._succeeded is False


@pytest.mark.asyncio
async def test_success_framework_evaluation_stops_without_response() -> None:
    opt = await init_optimizer(max_attempts=2)
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    end = await opt.on_event(RunEndEvent(evaluation=success_eval(), security_domain=USER_TAG))

    assert end.done is True
    assert opt._succeeded is True


@pytest.mark.asyncio
async def test_blind_scope_runs_full_retry_budget_without_guessing_success() -> None:
    opt = await init_optimizer(max_attempts=2)

    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    second = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert first.done is False
    assert second.done is True
    assert opt._succeeded is False


@pytest.mark.asyncio
async def test_no_signal_guard_can_stop_blind_scopes() -> None:
    opt = await init_optimizer(max_attempts=2, max_no_signal_runs=1)
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    end = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is True
    assert opt._stop_due_to_no_signal is True


@pytest.mark.asyncio
async def test_official_retry_adjustments_are_applied_from_visible_response() -> None:
    opt = await init_optimizer(max_attempts=3)
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    first_pre = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )
    assert isinstance(first_pre, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable(),
            request=first_pre.value,
            answer="I'm sorry, I cannot help with alpha beta",
        )
    )
    await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert opt._toxic_trunc == pytest.approx(0.4)

    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    second_pre = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )
    assert isinstance(second_pre, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable(),
            request=second_pre.value,
            answer="unrelated detailed answer",
        )
    )
    await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert opt._benign_trunc == pytest.approx(0.6)
