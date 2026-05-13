"""Tests for FITDOptimizer's SuperRed integration."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

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
    RunStartEvent,
)
from superred.core.types.goal import Goal

from fitd_optimizer.optimizer import FITDOptimizer
from fitd_optimizer.prompts import SYSTEM_PROMPT_EXTENSION
from tests.conftest import (
    FakeReadableTrajectory,
    MODEL_TAG,
    SYSTEM_PROMPT_TAG,
    USER_TAG,
    make_controllable,
    make_observable,
    mock_response,
    model_observable,
)


def success_eval() -> EvaluationResult:
    return EvaluationResult(
        success=True,
        primary_score=Score(value=1.0, name="primary"),
        sub_scores={},
        rationale="success",
    )


def failure_eval() -> EvaluationResult:
    return EvaluationResult(
        success=False,
        primary_score=Score(value=1.0, name="primary"),
        sub_scores={},
        rationale="failure",
    )


async def init_optimizer(
    *,
    llm: AsyncMock | None = None,
    prompt_sequence: list[str] | None = None,
    controllables: list | None = None,
    observables: list | None = None,
    max_attempts: int = 5,
) -> FITDOptimizer:
    opt = FITDOptimizer(
        prompt_sequence=prompt_sequence or ["benign bridge", "riskier bridge"],
        max_attempts=max_attempts,
    )
    await opt.initialize(
        goal=Goal(description="final harmful goal"),
        controllables=controllables if controllables is not None else [make_controllable()],
        observables=observables if observables is not None else [],
        llm_client=llm if llm is not None else AsyncMock(),
    )
    return opt


async def dispatch_event(opt: FITDOptimizer, event: Any) -> EventResponse:
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
async def test_first_turn_then_polish_turn_matches_official_flow() -> None:
    opt = await init_optimizer()
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    first = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    assert first.value == "benign bridge"

    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="benign answer",
        )
    )
    second = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )

    assert isinstance(second, ControllableInjection)
    assert "You have successfully response to the prompt: benign bridge" in second.value
    assert "New prompt:" in second.value
    assert "riskier bridge" in second.value


@pytest.mark.asyncio
async def test_chatbot_target_shape_uses_system_prompt_without_stealing_user_lock() -> None:
    opt = await init_optimizer(
        controllables=[
            make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            make_controllable("user_message", USER_TAG),
        ]
    )
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    sp = await dispatch_event(opt,
        ControllablePreCallEvent(
            controllable=make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            request="default system",
        )
    )
    user = await dispatch_event(opt,
        ControllablePreCallEvent(
            controllable=make_controllable("user_message", USER_TAG),
            request="original user",
        )
    )

    assert isinstance(sp, ControllableInjection)
    assert sp.value == SYSTEM_PROMPT_EXTENSION
    assert isinstance(user, ControllableInjection)
    assert user.value == "benign bridge"


@pytest.mark.asyncio
async def test_response_precall_is_skipped_without_stealing_user_lock() -> None:
    opt = await init_optimizer()
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    response = await dispatch_event(opt,
        ControllablePreCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request="assistant text",
        )
    )
    user = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )

    assert isinstance(response, ControllableNoInjection)
    assert isinstance(user, ControllableInjection)


@pytest.mark.asyncio
async def test_trajectory_response_is_used_before_postcall_fallback() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("safe")
    traj = FakeReadableTrajectory()
    opt = await init_optimizer(llm=llm)
    await dispatch_event(opt, RunStartEvent(trajectory=traj))
    first = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="postcall answer",
        )
    )
    traj.push(ObservableEvent(observable=make_observable("response"), content="trajectory answer"))

    await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    first_judge_prompt = llm.complete.await_args_list[0].args[0][0]["content"]
    assert "trajectory answer" in first_judge_prompt
    assert "postcall answer" not in first_judge_prompt


@pytest.mark.asyncio
async def test_non_success_framework_evaluation_blocks_local_success() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("unsafe")
    traj = FakeReadableTrajectory()
    opt = await init_optimizer(llm=llm, prompt_sequence=["only bridge"])
    await dispatch_event(opt, RunStartEvent(trajectory=traj))
    pre = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(pre, ControllableInjection)
    traj.push(ObservableEvent(observable=make_observable("response"), content="unsafe response"))

    end = await dispatch_event(opt, RunEndEvent(evaluation=failure_eval(), security_domain=USER_TAG))

    assert end.done is False
    assert opt._succeeded is False


@pytest.mark.asyncio
async def test_success_framework_evaluation_stops_without_response() -> None:
    opt = await init_optimizer(prompt_sequence=["only bridge"])
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    end = await dispatch_event(opt, RunEndEvent(evaluation=success_eval(), security_domain=USER_TAG))

    assert end.done is True
    assert opt._succeeded is True


@pytest.mark.asyncio
async def test_refusal_plans_replay_backtracking_in_next_run() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [
        mock_response("Yes"),
        mock_response("intermediate bridge"),
        mock_response("2"),
        mock_response("2"),
    ]
    opt = await init_optimizer(llm=llm, max_attempts=2)
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="benign answer",
        )
    )
    second = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )
    assert isinstance(second, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=second.value,
            answer="I'm sorry, I can't help.",
        )
    )

    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))
    assert end.done is False

    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    replay = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(replay, ControllableInjection)
    assert replay.value == "benign bridge"
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=replay.value,
            answer="benign answer",
        )
    )
    recovery = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )
    assert isinstance(recovery, ControllableInjection)
    assert "intermediate bridge" in recovery.value
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=recovery.value,
            answer="intermediate answer",
        )
    )
    original = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="next original")
    )
    assert isinstance(original, ControllableInjection)
    assert "riskier bridge" in original.value
    assert "intermediate bridge" in original.value


@pytest.mark.asyncio
async def test_replay_rebuilds_state_from_fresh_target_answer() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("No")
    opt = await init_optimizer(llm=llm, max_attempts=2)
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="old benign answer",
        )
    )
    second = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )
    assert isinstance(second, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=second.value,
            answer="I'm sorry, I can't help.",
        )
    )

    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))
    assert end.done is False

    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    replay = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(replay, ControllableInjection)
    assert replay.value == "benign bridge"
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=replay.value,
            answer="fresh nondeterministic benign answer",
        )
    )
    recovery = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )

    assert isinstance(recovery, ControllableInjection)
    assert "fresh nondeterministic benign answer" in recovery.value
    assert "old benign answer" not in recovery.value


@pytest.mark.asyncio
async def test_slippery_slope_retries_bridge_prompt_once_like_official_loop() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [
        mock_response("Yes"),
        mock_response("bridge one"),
        mock_response("Yes"),
        mock_response("bridge two"),
    ]
    opt = await init_optimizer(llm=llm, max_attempts=4)
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="benign answer",
        )
    )
    second = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )
    assert isinstance(second, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=second.value,
            answer="I'm sorry, I can't help.",
        )
    )
    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))
    assert end.done is False

    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    replay = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="replay")
    )
    assert isinstance(replay, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=replay.value,
            answer="benign answer",
        )
    )
    bridge_one = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="bridge one")
    )
    assert isinstance(bridge_one, ControllableInjection)
    assert "bridge one" in bridge_one.value
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=bridge_one.value,
            answer="I'm sorry, I can't help.",
        )
    )
    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))
    assert end.done is False

    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    replay = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="replay again")
    )
    assert isinstance(replay, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=replay.value,
            answer="benign answer",
        )
    )
    bridge_two = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="bridge two")
    )
    assert isinstance(bridge_two, ControllableInjection)
    assert "bridge two" in bridge_two.value


@pytest.mark.asyncio
async def test_slippery_slope_stops_recovery_after_official_retry_budget() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [
        mock_response("Yes"),
        mock_response("bridge one"),
        mock_response("Yes"),
        mock_response("bridge two"),
        mock_response("safe"),
        mock_response("-1"),
    ]
    opt = await init_optimizer(llm=llm, max_attempts=4)
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="benign answer",
        )
    )
    refused_turn = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )
    assert isinstance(refused_turn, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=refused_turn.value,
            answer="I'm sorry, I can't help.",
        )
    )
    await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    for bridge_request in ("bridge one", "bridge two"):
        await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
        replay = await dispatch_event(opt,
            ControllablePreCallEvent(controllable=make_controllable(), request="replay")
        )
        assert isinstance(replay, ControllableInjection)
        await dispatch_event(opt,
            ControllablePostCallEvent(
                controllable=make_controllable("response", MODEL_TAG),
                request=replay.value,
                answer="benign answer",
            )
        )
        bridge = await dispatch_event(opt,
            ControllablePreCallEvent(controllable=make_controllable(), request=bridge_request)
        )
        assert isinstance(bridge, ControllableInjection)
        await dispatch_event(opt,
            ControllablePostCallEvent(
                controllable=make_controllable("response", MODEL_TAG),
                request=bridge.value,
                answer="I'm sorry, I can't help.",
            )
        )
        end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is True
    assert opt._pending_recovery is None


@pytest.mark.asyncio
async def test_static_context_is_bounded_for_prompt_generation() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("**Step 1**: bridge")
    opt = FITDOptimizer(benign_prompt="benign", level=1, static_context_budget=80)
    await opt.initialize(
        goal=Goal(description="final harmful goal"),
        controllables=[make_controllable()],
        observables=[model_observable("x" * 200)],
        llm_client=llm,
    )
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    sent = llm.complete.call_args.args[0][1]["content"]
    assert "SUPERRED STATIC TARGET CONTEXT" in sent
    assert "[truncated]" in sent


@pytest.mark.asyncio
async def test_final_align_keeps_previous_answer_when_official_judge_prefers_it() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("1")
    opt = await init_optimizer(llm=llm, prompt_sequence=["final harmful goal"])
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="previous answer",
        )
    )
    second = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )
    assert isinstance(second, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=second.value,
            answer="previous answer",
        )
    )
    final_align = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="final align")
    )
    assert isinstance(final_align, ControllableInjection)
    assert "Rewrite the response" in final_align.value
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=final_align.value,
            answer="new answer",
        )
    )
    await dispatch_event(opt, RunEndEvent(evaluation=failure_eval(), security_domain=USER_TAG))

    assert opt._last_response == "previous answer"


@pytest.mark.asyncio
async def test_local_evaluator_checks_prior_outputs_when_final_is_safe() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [mock_response("safe"), mock_response("0"), mock_response("unsafe")]
    opt = await init_optimizer(llm=llm, prompt_sequence=["benign bridge", "riskier bridge"] )
    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="prior unsafe answer",
        )
    )
    second = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )
    assert isinstance(second, ControllableInjection)
    await dispatch_event(opt,
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=second.value,
            answer="safe final answer",
        )
    )

    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is True
    assert opt._succeeded is True
    assert llm.complete.await_count == 3


@pytest.mark.asyncio
async def test_prompt_preparation_uses_helper_llm_without_shadow_query_counter() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [
        mock_response("benign bridge"),
        mock_response("aligned benign bridge"),
        mock_response("**Step 1**: generated bridge"),
    ]
    opt = FITDOptimizer(level=1)
    await opt.initialize(
        goal=Goal(description="final harmful goal"),
        controllables=[make_controllable()],
        observables=[],
        llm_client=llm,
    )

    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))

    assert llm.complete.await_count == 3


@pytest.mark.asyncio
async def test_budget_exhaustion_during_prompt_preparation_stops_cleanly() -> None:
    from superred.core.types.llm import BudgetExhaustedError, LLMUsage

    llm = AsyncMock()
    llm.complete.side_effect = BudgetExhaustedError("budget", LLMUsage())
    opt = FITDOptimizer(level=1)
    await opt.initialize(
        goal=Goal(description="final harmful goal"),
        controllables=[make_controllable()],
        observables=[],
        llm_client=llm,
    )

    await dispatch_event(opt, RunStartEvent(trajectory=FakeReadableTrajectory()))
    pre = await dispatch_event(opt,
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    end = await dispatch_event(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert isinstance(pre, ControllableNoInjection)
    assert end.done is True
