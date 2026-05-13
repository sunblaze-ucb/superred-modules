"""Tests for PAIROptimizer's SuperRed integration."""

from __future__ import annotations

from unittest.mock import AsyncMock

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
from superred.core.types.llm import BudgetExhaustedError, LLMUsage

from pair_optimizer.optimizer import PAIROptimizer
from tests.conftest import (
    FakeReadableTrajectory,
    MODEL_TAG,
    SYSTEM_PROMPT_READABLE_TAG,
    SYSTEM_PROMPT_TAG,
    USER_TAG,
    make_controllable,
    make_observable,
    mock_response,
    observable_value,
    dispatch_event,
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
    llm: AsyncMock | None = None,
    n_streams: int = 3,
    n_iterations: int = 3,
    judge_method: str = "gcg",
) -> PAIROptimizer:
    opt = PAIROptimizer(n_streams=n_streams, n_iterations=n_iterations, judge_method=judge_method)
    await opt.initialize(
        goal=Goal(description="Give detailed instructions"),
        controllables=controllables if controllables is not None else [make_controllable()],
        observables=observables if observables is not None else [],
        llm_client=llm if llm is not None else AsyncMock(),
    )
    return opt


@pytest.mark.asyncio
async def test_defaults_match_official_runtime_defaults() -> None:
    opt = PAIROptimizer()
    assert opt.n_streams == 3
    assert opt.n_iterations == 3
    assert opt.keep_last_n == 4
    assert opt.attack_max_tokens == 500
    assert opt.max_attack_attempts == 5
    assert opt.judge_max_tokens == 10
    assert opt.judge_method == "gcg"


@pytest.mark.asyncio
async def test_generates_one_candidate_per_stream_before_advancing_iteration() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [
        mock_response('{"improvement": "one", "prompt": "attack one"}'),
        mock_response('{"improvement": "two", "prompt": "attack two"}'),
    ]
    opt = await init_optimizer(llm=llm, n_streams=2, n_iterations=1)

    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="u")
    )
    end1 = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    second = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="u")
    )
    end2 = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert isinstance(first, ControllableInjection)
    assert isinstance(second, ControllableInjection)
    assert first.value == "attack one"
    assert second.value == "attack two"
    assert isinstance(end1, RunEndResponse) and end1.done is False
    assert isinstance(end2, RunEndResponse) and end2.done is True
    assert llm.complete.call_count == 2


@pytest.mark.asyncio
async def test_system_prompt_precall_does_not_steal_user_lock_when_not_writable() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response(
        '{"improvement": "better", "prompt": "this attack prompt has enough words"}'
    )
    opt = await init_optimizer(llm=llm, controllables=[make_controllable("user_message", USER_TAG)])
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    sp = await dispatch_event(opt,
        ControllablePreCallEvent(
            controllable=make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            request="default",
        )
    )
    user = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )

    assert isinstance(sp, ControllableNoInjection)
    assert isinstance(user, ControllableInjection)
    assert user.value == "this attack prompt has enough words"


@pytest.mark.asyncio
async def test_writable_system_prompt_extension_injects_non_empty_system_prompt() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response(
        '{"improvement": "better", "prompt": "attack", "system_prompt": "override"}'
    )
    opt = await init_optimizer(
        llm=llm,
        controllables=[
            make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            make_controllable("user_message", USER_TAG),
        ],
    )
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    sp = await dispatch_event(opt,
        ControllablePreCallEvent(
            controllable=make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            request="default",
        )
    )
    user = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )

    assert isinstance(sp, ControllableInjection)
    assert sp.value == "override"
    assert isinstance(user, ControllableInjection)
    assert user.value == "attack"


@pytest.mark.asyncio
async def test_blank_system_prompt_extension_is_skipped() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response(
        '{"improvement": "better", "prompt": "attack", "system_prompt": ""}'
    )
    opt = await init_optimizer(
        llm=llm,
        controllables=[
            make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            make_controllable("user_message", USER_TAG),
        ],
    )
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    sp = await dispatch_event(opt,
        ControllablePreCallEvent(
            controllable=make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            request="default",
        )
    )
    user = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )

    assert isinstance(sp, ControllableNoInjection)
    assert isinstance(user, ControllableInjection)


@pytest.mark.asyncio
async def test_response_precall_is_skipped_without_stealing_user_lock() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response('{"improvement": "better", "prompt": "attack"}')
    opt = await init_optimizer(llm=llm)
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    response_pre = await dispatch_event(opt,
        ControllablePreCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request="assistant answer",
        )
    )
    user = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )

    assert isinstance(response_pre, ControllableNoInjection)
    assert isinstance(user, ControllableInjection)


@pytest.mark.asyncio
async def test_trajectory_response_is_used_before_postcall_fallback() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response(
        '{"improvement": "better", "prompt": "this attack prompt has enough words"}'
    )
    traj = FakeReadableTrajectory()
    opt = await init_optimizer(llm=llm, n_streams=1, n_iterations=1, judge_method="gcg")
    await dispatch_event(opt, RunStartEvent(trajectory=traj))
    pre = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="u")
    )
    assert isinstance(pre, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=pre.value,
            answer="I'm sorry, postcall refused",
        )
    )
    traj.push(
        ObservableEvent(
            observable=make_observable("response"),
            content="Sure here are many detailed words in a trajectory answer",
        )
    )

    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert isinstance(end, RunEndResponse)
    assert end.done is True
    assert opt._best_score == 10
    assert (
        opt._streams[0].last_response == "Sure here are many detailed words in a trajectory answer"
    )


@pytest.mark.asyncio
async def test_framework_failure_caps_local_judge_success_when_response_visible() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response(
        '{"improvement": "better", "prompt": "this attack prompt has enough words"}'
    )
    traj = FakeReadableTrajectory()
    opt = await init_optimizer(llm=llm, n_streams=1, n_iterations=2, judge_method="gcg")
    await dispatch_event(opt, RunStartEvent(trajectory=traj))
    await dispatch_event(opt, ControllablePreCallEvent(controllable=make_controllable(), request="u"))
    traj.push(
        ObservableEvent(
            observable=make_observable("response"),
            content="Sure here are many detailed words in a trajectory answer",
        )
    )

    end = await dispatch_event(
        opt, RunEndEvent(evaluation=failure_eval(), security_domain=USER_TAG)
    )

    assert isinstance(end, RunEndResponse)
    assert end.done is False
    assert opt._succeeded is False
    assert opt._streams[0].last_score == 9.0
    assert "FRAMEWORK EVALUATION RATIONALE: task failed" in opt._streams[0].last_response


@pytest.mark.asyncio
async def test_framework_failure_with_max_primary_score_does_not_count_as_success() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response('{"improvement": "better", "prompt": "attack"}')
    opt = await init_optimizer(llm=llm, n_streams=1, n_iterations=2)
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    await dispatch_event(opt, ControllablePreCallEvent(controllable=make_controllable(), request="u"))

    end = await dispatch_event(opt, RunEndEvent(evaluation=failure_eval(), security_domain=USER_TAG))

    assert end.done is False
    assert opt._succeeded is False
    assert opt._streams[0].last_score < 10


@pytest.mark.asyncio
async def test_framework_success_is_authoritative_without_response() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response('{"improvement": "better", "prompt": "attack"}')
    opt = await init_optimizer(llm=llm, n_streams=1, n_iterations=3)
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    await dispatch_event(opt, ControllablePreCallEvent(controllable=make_controllable(), request="u"))

    end = await dispatch_event(opt, RunEndEvent(evaluation=success_eval(), security_domain=USER_TAG))

    assert end.done is True
    assert opt._succeeded is True
    assert opt._best_score == 10


@pytest.mark.asyncio
async def test_static_context_is_added_to_attacker_prompt_with_budget() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response('{"improvement": "better", "prompt": "attack"}')
    opt = await init_optimizer(
        llm=llm,
        observables=[
            observable_value("model", "target-model"),
            observable_value("system_prompt", "readable system" * 500, SYSTEM_PROMPT_READABLE_TAG),
        ],
    )

    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    attacker_system = llm.complete.call_args.args[0][0]["content"]
    assert "SUPERRED STATIC TARGET CONTEXT" in attacker_system
    assert "target-model" in attacker_system
    assert len(attacker_system) < 24000


@pytest.mark.asyncio
async def test_budget_errors_propagate_from_attacker_generation() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = BudgetExhaustedError("budget", usage=LLMUsage(calls=1, cost=1.0))
    opt = await init_optimizer(llm=llm)

    with pytest.raises(BudgetExhaustedError):
        await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
