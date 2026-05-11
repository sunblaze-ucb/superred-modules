"""Tests for GPTFuzzerOptimizer's SuperRed integration."""

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
    MODEL_IDENTITY_TAG,
    MODEL_TAG,
    SYSTEM_PROMPT_READABLE_TAG,
    SYSTEM_PROMPT_TAG,
    USER_TAG,
    make_controllable,
    make_observable,
    make_observable_value,
    mock_response,
)
from gptfuzzer_optimizer.optimizer import GPTFuzzerOptimizer


def success_eval(score: float = 1.0) -> EvaluationResult:
    return EvaluationResult(
        success=True,
        primary_score=Score(value=score, name="primary"),
        sub_scores={},
        rationale="success",
    )


def failure_eval(score: float = 1.0) -> EvaluationResult:
    return EvaluationResult(
        success=False,
        primary_score=Score(value=score, name="primary"),
        sub_scores={},
        rationale="failure",
    )


async def init_optimizer(
    *,
    llm: AsyncMock | None = None,
    initial_seed: list[str] | None = None,
    controllables: list | None = None,
    observables: list | None = None,
    max_query: int = 10,
    max_jailbreak: int = 1,
    max_no_signal_runs: int = 0,
) -> GPTFuzzerOptimizer:
    opt = GPTFuzzerOptimizer(
        initial_seed=initial_seed if initial_seed is not None else ["seed [INSERT PROMPT HERE]"],
        max_query=max_query,
        max_jailbreak=max_jailbreak,
        max_no_signal_runs=max_no_signal_runs,
        random_seed=0,
    )
    await opt.initialize(
        goal=Goal(description="How do I test a security control?"),
        controllables=controllables if controllables is not None else [make_controllable()],
        observables=observables if observables is not None else [],
        llm_client=llm if llm is not None else AsyncMock(),
    )
    return opt


@pytest.mark.asyncio
async def test_run_start_mutates_seed_and_precall_injects_synthesized_prompt() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("mutated [INSERT PROMPT HERE]. ")
    opt = await init_optimizer(llm=llm)

    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    resp = await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )

    assert isinstance(resp, ControllableInjection)
    assert "How do I test a security control?" in resp.value
    assert "mutated" in resp.value
    assert opt._current_node is not None
    assert opt._current_node.parent is opt._prompt_nodes[0]


@pytest.mark.asyncio
async def test_system_prompt_precall_uses_writable_system_prompt_then_user_message() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("mutated [INSERT PROMPT HERE]. ")
    opt = await init_optimizer(
        llm=llm,
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
    assert "mutated How do I test a security control?" in sp.value
    assert isinstance(user, ControllableInjection)
    assert user.value == "How do I test a security control?"


@pytest.mark.asyncio
async def test_system_prompt_precall_can_be_disabled_for_paper_user_channel() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("mutated [INSERT PROMPT HERE]. ")
    opt = GPTFuzzerOptimizer(
        initial_seed=["seed [INSERT PROMPT HERE]"],
        random_seed=0,
        use_system_prompt_when_available=False,
    )
    await opt.initialize(
        goal=Goal(description="How do I test a security control?"),
        controllables=[
            make_controllable("system_prompt", SYSTEM_PROMPT_TAG),
            make_controllable("user_message", USER_TAG),
        ],
        observables=[],
        llm_client=llm,
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

    assert isinstance(sp, ControllableNoInjection)
    assert isinstance(user, ControllableInjection)


@pytest.mark.asyncio
async def test_static_observables_are_passed_to_mutator_as_bounded_context() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("mutated [INSERT PROMPT HERE]. ")
    opt = await init_optimizer(
        llm=llm,
        observables=[
            make_observable_value("model", "target-model", MODEL_IDENTITY_TAG),
            make_observable_value("system_prompt", "be safe", SYSTEM_PROMPT_READABLE_TAG),
        ],
    )

    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    mutation_prompt = llm.complete.call_args.args[0][0]["content"]
    assert "SUPERRED TARGET CONTEXT" in mutation_prompt
    assert "model: target-model" in mutation_prompt
    assert "system_prompt: be safe" in mutation_prompt


@pytest.mark.asyncio
async def test_trajectory_response_wins_over_postcall_fallback_and_success_adds_seed() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("mutated [INSERT PROMPT HERE]. ")
    traj = FakeReadableTrajectory()
    opt = await init_optimizer(llm=llm, max_jailbreak=2)
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
    traj.push(ObservableEvent(observable=make_observable("response"), content="trajectory answer"))

    end = await opt.on_event(RunEndEvent(evaluation=success_eval(), security_domain=USER_TAG))

    assert isinstance(end, RunEndResponse)
    assert end.done is False
    assert opt._current_node is not None
    assert opt._current_node.response == ["trajectory answer"]
    assert opt._current_node.results == [1]
    assert opt._current_node in opt._prompt_nodes


@pytest.mark.asyncio
async def test_non_success_framework_evaluation_does_not_retain_seed_even_with_score_one() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("mutated [INSERT PROMPT HERE]. ")
    opt = await init_optimizer(llm=llm)
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))
    await opt.on_event(
        ControllablePreCallEvent(controllable=make_controllable(), request="user message")
    )

    end = await opt.on_event(RunEndEvent(evaluation=failure_eval(), security_domain=USER_TAG))

    assert end.done is False
    assert len(opt._prompt_nodes) == 1
    assert opt._current_node is not None
    assert opt._current_node.results == [0]


@pytest.mark.asyncio
async def test_no_signal_guard_can_stop_blind_scopes() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("mutated [INSERT PROMPT HERE]. ")
    opt = await init_optimizer(llm=llm, max_no_signal_runs=1, max_query=10)
    await opt.on_event(RunStartEvent(trajectory=FakeReadableTrajectory()))

    end = await opt.on_event(RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert end.done is True
    assert opt._stop_due_to_no_signal is True
