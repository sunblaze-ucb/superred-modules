"""Tests for FITDOptimizer's SuperRed integration."""

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


@pytest.mark.asyncio
async def test_first_turn_then_polish_turn_matches_official_flow() -> None:
    opt = await init_optimizer()
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    first = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    assert first.value == "benign bridge"

    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="benign answer",
        )
    )
    second = await opt.on_event(
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
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    sp = await opt.on_event(
        ControllablePreCallEvent(
            controllable=make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            request="default system",
        )
    )
    user = await opt.on_event(
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
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    response = await opt.on_event(
        ControllablePreCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request="assistant text",
        )
    )
    user = await opt.on_event(
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
    await opt.on_event(RunStartEvent(trajectory=traj))
    first = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="postcall answer",
        )
    )
    traj.push(ObservableEvent(observable=make_observable("response"), content="trajectory answer"))

    await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    first_judge_prompt = llm.complete.await_args_list[0].args[0][0]["content"]
    assert "trajectory answer" in first_judge_prompt
    assert "postcall answer" not in first_judge_prompt


@pytest.mark.asyncio
async def test_non_success_framework_evaluation_blocks_local_success() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("unsafe")
    traj = FakeReadableTrajectory()
    opt = await init_optimizer(llm=llm, prompt_sequence=["only bridge"])
    await opt.on_event(RunStartEvent(trajectory=traj))
    pre = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(pre, ControllableInjection)
    traj.push(ObservableEvent(observable=make_observable("response"), content="unsafe response"))

    end = await opt.on_event(RunEndEvent(evaluation=failure_eval(), security_domain=USER_TAG))

    assert end.done is False
    assert opt._succeeded is False


@pytest.mark.asyncio
async def test_success_framework_evaluation_stops_without_response() -> None:
    opt = await init_optimizer(prompt_sequence=["only bridge"])
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    end = await opt.on_event(RunEndEvent(evaluation=success_eval(), security_domain=USER_TAG))

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
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="benign answer",
        )
    )
    second = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )
    assert isinstance(second, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=second.value,
            answer="I'm sorry, I can't help.",
        )
    )

    end = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))
    assert end.done is False

    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    replay = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(replay, ControllableInjection)
    assert replay.value == "benign bridge"
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=replay.value,
            answer="benign answer",
        )
    )
    recovery = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )
    assert isinstance(recovery, ControllableInjection)
    assert "intermediate bridge" in recovery.value
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=recovery.value,
            answer="intermediate answer",
        )
    )
    original = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="next original")
    )
    assert isinstance(original, ControllableInjection)
    assert "riskier bridge" in original.value
    assert "intermediate bridge" in original.value


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
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    sent = llm.complete.call_args.args[0][1]["content"]
    assert "SUPERRED STATIC TARGET CONTEXT" in sent
    assert "[truncated]" in sent


@pytest.mark.asyncio
async def test_final_align_keeps_previous_answer_when_official_judge_prefers_it() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("1")
    opt = await init_optimizer(llm=llm, prompt_sequence=["final harmful goal"])
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="previous answer",
        )
    )
    second = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )
    assert isinstance(second, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=second.value,
            answer="previous answer",
        )
    )
    final_align = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="final align")
    )
    assert isinstance(final_align, ControllableInjection)
    assert "Rewrite the response" in final_align.value
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=final_align.value,
            answer="new answer",
        )
    )
    await opt.on_event(RunEndEvent(evaluation=failure_eval(), security_domain=USER_TAG))

    assert opt._last_response == "previous answer"


@pytest.mark.asyncio
async def test_local_evaluator_checks_prior_outputs_when_final_is_safe() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [mock_response("safe"), mock_response("0"), mock_response("unsafe")]
    opt = await init_optimizer(llm=llm, prompt_sequence=["benign bridge", "riskier bridge"] )
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="original")
    )
    assert isinstance(first, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=first.value,
            answer="prior unsafe answer",
        )
    )
    second = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="next")
    )
    assert isinstance(second, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=second.value,
            answer="safe final answer",
        )
    )

    end = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is True
    assert opt._succeeded is True
    assert llm.complete.await_count == 3
