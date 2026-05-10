"""Tests for CodeChameleonOptimizer's SuperRed integration."""

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
from code_chameleon_optimizer.optimizer import CodeChameleonOptimizer
from code_chameleon_optimizer.prompts import CODE_SYSTEM_PROMPT


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
    encrypt_rules: list[str] | None = None,
    max_no_signal_runs: int = 0,
) -> CodeChameleonOptimizer:
    opt = CodeChameleonOptimizer(
        encrypt_rules=encrypt_rules,
        max_no_signal_runs=max_no_signal_runs,
    )
    await opt.initialize(
        goal=Goal(description="Give detailed instructions"),
        controllables=controllables if controllables is not None else [make_controllable()],
        observables=observables if observables is not None else [],
        llm_client=llm if llm is not None else AsyncMock(),
    )
    return opt


@pytest.mark.asyncio
async def test_default_rule_is_binary_tree_without_model_observable() -> None:
    opt = await init_optimizer()
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    assert opt._current_encrypt_rule == "binary_tree"
    assert "The encrypted_problem you need to solve is:" in opt._current_user_prompt


@pytest.mark.asyncio
async def test_model_observable_can_select_paper_table_rule_when_not_explicit() -> None:
    opt = await init_optimizer(observables=[model_observable("Llama-2-70b-chat")])
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    assert opt._current_encrypt_rule == "length"


@pytest.mark.asyncio
async def test_explicit_rule_wins_over_model_observable() -> None:
    opt = await init_optimizer(
        encrypt_rules=["reverse"],
        observables=[model_observable("Llama-2-70b-chat")],
    )
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    assert opt._current_encrypt_rule == "reverse"


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
    assert sp.value == CODE_SYSTEM_PROMPT
    assert isinstance(user, ControllableInjection)
    assert CODE_SYSTEM_PROMPT not in user.value
    assert "ProblemSolver" in user.value


@pytest.mark.asyncio
async def test_system_prompt_precall_does_not_steal_lock_when_not_writable() -> None:
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

    response_ctrl = make_controllable("response", MODEL_TAG)
    response_pre = await opt.on_event(
        ControllablePreCallEvent(controllable=response_ctrl, request="assistant answer")
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
async def test_trajectory_response_is_scored_before_postcall_fallback() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("#thescore: 5")
    traj = FakeReadableTrajectory()
    opt = await init_optimizer(llm=llm)
    await opt.on_event(RunStartEvent(trajectory=traj))
    pre = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )
    assert isinstance(pre, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=pre.value,
            answer="postcall answer",
        )
    )
    traj.push(
        ObservableEvent(
            observable=make_observable("response"),
            content="trajectory answer",
        )
    )

    end = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert isinstance(end, RunEndResponse)
    assert end.done is True
    assert "trajectory answer" in llm.complete.call_args.args[0][0]["content"]
    assert "postcall answer" not in llm.complete.call_args.args[0][0]["content"]


@pytest.mark.asyncio
async def test_postcall_answer_is_used_when_trajectory_response_is_not_visible() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("#thescore: 5")
    opt = await init_optimizer(llm=llm)
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    pre = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )
    assert isinstance(pre, ControllableInjection)
    await opt.on_event(
        ControllablePostCallEvent(
            controllable=make_controllable("response", MODEL_TAG),
            request=pre.value,
            answer="postcall answer",
        )
    )

    end = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is True
    assert "postcall answer" in llm.complete.call_args.args[0][0]["content"]


@pytest.mark.asyncio
async def test_non_success_framework_evaluation_blocks_internal_success() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("#thescore: 5")
    traj = FakeReadableTrajectory()
    traj.push(ObservableEvent(observable=make_observable("response"), content="unsafe answer"))
    opt = await init_optimizer(llm=llm, encrypt_rules=["reverse", "binary_tree"])
    await opt.on_event(RunStartEvent(trajectory=traj))
    await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )

    end = await opt.on_event(RunEndEvent(evaluation=failure_eval(), security_domain=USER_TAG))

    assert end.done is False
    assert opt._succeeded is False


@pytest.mark.asyncio
async def test_success_framework_evaluation_stops_without_response() -> None:
    opt = await init_optimizer(encrypt_rules=["reverse", "binary_tree"])
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    end = await opt.on_event(RunEndEvent(evaluation=success_eval(), security_domain=USER_TAG))

    assert end.done is True
    assert opt._succeeded is True


@pytest.mark.asyncio
async def test_blind_scope_runs_configured_rule_schedule_without_guessing_success() -> None:
    opt = await init_optimizer(encrypt_rules=["reverse", "binary_tree"])

    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    first = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    second = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert first.done is False
    assert second.done is True
    assert opt._succeeded is False


@pytest.mark.asyncio
async def test_no_signal_guard_can_stop_blind_scopes() -> None:
    opt = await init_optimizer(encrypt_rules=["reverse", "binary_tree"], max_no_signal_runs=1)
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    end = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is True
    assert opt._stop_due_to_no_signal is True
