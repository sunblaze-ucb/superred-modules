"""Tests for DraOptimizer's SuperRed integration."""

from __future__ import annotations

import asyncio
from typing import Any
from random import Random

import pytest

from superred.core.channel import EventEnvelope
from superred.core.types.event import EventResponse
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

import dra_optimizer.optimizer as optimizer_module
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
    judge_response: str = "Rating: [[10]]",
    use_llm_judge_when_no_feedback: bool = True,
) -> DraOptimizer:
    opt = DraOptimizer(
        max_attempts=max_attempts,
        max_no_signal_runs=max_no_signal_runs,
        model=model,
        rng=Random(9),
        use_llm_judge_when_no_feedback=use_llm_judge_when_no_feedback,
    )
    await opt.initialize(
        goal=Goal(description="alpha beta"),
        controllables=controllables if controllables is not None else [make_controllable()],
        observables=observables if observables is not None else [],
        llm_client=fake_llm(judge_response),
    )
    return opt


async def init_optimizer_with_llm(
    llm,
    *,
    max_attempts: int = 2,
) -> DraOptimizer:
    opt = DraOptimizer(max_attempts=max_attempts, rng=Random(9))
    await opt.initialize(
        goal=Goal(description="alpha beta"),
        controllables=[make_controllable()],
        observables=[],
        llm_client=llm,
    )
    return opt


async def dispatch_event(opt: DraOptimizer, event: Any) -> EventResponse:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[EventResponse] = loop.create_future()
    envelope = EventEnvelope(event=event, future=future, loop=loop)
    try:
        await opt._dispatch(envelope)
    except Exception:
        await asyncio.sleep(0)
        if future.done():
            future.exception()
        raise
    return await future


@pytest.mark.asyncio
async def test_default_model_is_llama_without_model_observable() -> None:
    opt = await init_optimizer()
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    assert opt._model == "llama"
    assert opt._current_attack is not None
    assert opt._current_attack.model == "llama"


@pytest.mark.asyncio
async def test_model_observable_selects_official_suffix_family() -> None:
    opt = await init_optimizer(observables=[model_observable("gpt-4o-2024-05-13")])
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    assert opt._model == "gpt-4o"
    assert opt._current_attack is not None
    assert opt._current_attack.model == "gpt-4o"


@pytest.mark.asyncio
async def test_default_sensitive_detector_prefers_detoxify_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def detector(token: str) -> bool:
        return token == "alpha"

    monkeypatch.setattr(
        optimizer_module,
        "try_create_detoxify_token_detector",
        lambda: detector,
    )

    opt = await init_optimizer()

    assert opt._prompt_builder is not None
    assert opt._prompt_builder._sensitive_token_detector is detector


@pytest.mark.asyncio
async def test_explicit_sensitive_detector_overrides_detoxify(monkeypatch: pytest.MonkeyPatch) -> None:
    def explicit_detector(token: str) -> bool:
        return token == "beta"

    monkeypatch.setattr(
        optimizer_module,
        "try_create_detoxify_token_detector",
        lambda: (lambda token: token == "alpha"),
    )
    opt = DraOptimizer(
        rng=Random(9),
        sensitive_token_detector=explicit_detector,
    )

    await opt.initialize(
        goal=Goal(description="alpha beta"),
        controllables=[make_controllable()],
        observables=[],
        llm_client=fake_llm("Rating: [[10]]"),
    )

    assert opt._prompt_builder is not None
    assert opt._prompt_builder._sensitive_token_detector is explicit_detector


@pytest.mark.asyncio
async def test_chatbot_target_shape_injects_system_prompt_then_user_prompt() -> None:
    opt = await init_optimizer(
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
async def test_system_prompt_only_scope_delivers_combined_attack_prompt() -> None:
    opt = await init_optimizer(
        controllables=[make_controllable("system_prompt", SYSTEM_PROMPT_TAG)],
    )
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    sp = await dispatch_event(
        opt,
        ControllablePreCallEvent(
            controllable=make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            request="default",
        ),
    )

    assert isinstance(sp, ControllableInjection)
    assert sp.value.startswith(OFFICIAL_SYSTEM_PROMPT)
    assert "construct a sentence named Q1" in sp.value


@pytest.mark.asyncio
async def test_system_prompt_precall_does_not_steal_user_lock_when_not_writable() -> None:
    opt = await init_optimizer(controllables=[make_controllable("user_message", USER_TAG)])
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    sp = await dispatch_event(opt,
        ControllablePreCallEvent(
            controllable=make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            request="default",
        )
    )
    user = await dispatch_event(opt,
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
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    response_pre = await dispatch_event(opt,
        ControllablePreCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request="assistant answer",
        )
    )
    user = await dispatch_event(opt,
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
    await dispatch_event(opt, RunStartEvent(trajectory=traj))
    pre = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )
    assert isinstance(pre, ControllableInjection)
    await dispatch_event(opt,
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

    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert isinstance(end, RunEndResponse)
    assert end.done is True
    assert opt._succeeded is True
    assert opt._last_check is not None
    assert opt._last_check.jailbroken_gcg is True


@pytest.mark.asyncio
async def test_postcall_answer_is_used_when_trajectory_response_is_not_visible() -> None:
    opt = await init_optimizer()
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    pre = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )
    assert isinstance(pre, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=pre.value,
            answer="alpha beta detailed answer",
        )
    )

    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is True
    assert opt._succeeded is True


@pytest.mark.asyncio
async def test_lightweight_success_requires_official_gpt_judge_success_by_default() -> None:
    opt = await init_optimizer(max_attempts=2, judge_response="Rating: [[9]]")
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    pre = await dispatch_event(
        opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="user message"),
    )
    assert isinstance(pre, ControllableInjection)
    await dispatch_event(
        opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=pre.value,
            answer="alpha beta detailed answer",
        ),
    )

    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is False
    assert opt._succeeded is False


@pytest.mark.asyncio
async def test_llm_judge_can_be_disabled_for_lightweight_only_fallback() -> None:
    opt = await init_optimizer(
        use_llm_judge_when_no_feedback=False,
        judge_response="Rating: [[1]]",
    )
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    pre = await dispatch_event(
        opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="user message"),
    )
    assert isinstance(pre, ControllableInjection)
    await dispatch_event(
        opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=pre.value,
            answer="alpha beta detailed answer",
        ),
    )

    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is True
    assert opt._succeeded is True


@pytest.mark.asyncio
async def test_failed_gpt_judge_does_not_mark_success() -> None:
    llm = fake_llm()
    llm.complete.side_effect = RuntimeError("judge transport failed")
    opt = await init_optimizer_with_llm(llm)
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    pre = await dispatch_event(
        opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="user message"),
    )
    assert isinstance(pre, ControllableInjection)
    await dispatch_event(
        opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=pre.value,
            answer="alpha beta detailed answer",
        ),
    )

    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is False
    assert opt._succeeded is False


@pytest.mark.asyncio
async def test_budget_exhausted_from_gpt_judge_propagates() -> None:
    llm = fake_llm()
    llm.complete.side_effect = BudgetExhaustedError("budget exhausted", LLMUsage())
    opt = await init_optimizer_with_llm(llm)
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    pre = await dispatch_event(
        opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="user message"),
    )
    assert isinstance(pre, ControllableInjection)
    await dispatch_event(
        opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=pre.value,
            answer="alpha beta detailed answer",
        ),
    )

    with pytest.raises(BudgetExhaustedError):
        await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))


@pytest.mark.asyncio
async def test_non_success_framework_evaluation_blocks_local_success() -> None:
    traj = FakeReadableTrajectory()
    traj.push(ObservableEvent(observable=make_observable("response"), content="alpha beta answer"))
    opt = await init_optimizer(max_attempts=2)
    await dispatch_event(opt, RunStartEvent(trajectory=traj))

    end = await dispatch_event(opt, RunEndEvent(evaluation=failure_eval(), security_domain=USER_TAG))

    assert end.done is False
    assert opt._succeeded is False
    assert opt.llm.complete.call_count == 0


@pytest.mark.asyncio
async def test_success_framework_evaluation_stops_without_response() -> None:
    opt = await init_optimizer(max_attempts=2)
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    end = await dispatch_event(opt, RunEndEvent(evaluation=success_eval(), security_domain=USER_TAG))

    assert end.done is True
    assert opt._succeeded is True


@pytest.mark.asyncio
async def test_blind_scope_runs_full_retry_budget_without_guessing_success() -> None:
    opt = await init_optimizer(max_attempts=2)

    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    second = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert first.done is False
    assert second.done is True
    assert opt._succeeded is False


@pytest.mark.asyncio
async def test_no_signal_guard_can_stop_blind_scopes() -> None:
    opt = await init_optimizer(max_attempts=2, max_no_signal_runs=1)
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is True
    assert opt._stop_due_to_no_signal is True


@pytest.mark.asyncio
async def test_official_retry_adjustments_are_applied_from_visible_response() -> None:
    opt = await init_optimizer(max_attempts=3)
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    first_pre = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )
    assert isinstance(first_pre, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable(),
            request=first_pre.value,
            answer="I'm sorry, I cannot help with alpha beta",
        )
    )
    await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert opt._toxic_trunc == pytest.approx(0.4)

    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    second_pre = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )
    assert isinstance(second_pre, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable(),
            request=second_pre.value,
            answer="unrelated detailed answer",
        )
    )
    await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert opt._benign_trunc == pytest.approx(0.6)
