"""Tests for FlipAttackOptimizer event-driven state machine."""

from unittest.mock import AsyncMock
import pytest

from superred.core.types.event import EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    ControllablePostCallEvent,
    ObservableEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import Observable
from superred.core.types.security_domain import SecurityDomainTag
from superred.core.types.trajectory import Trajectory

from tests.conftest import DOMAIN, mock_response, make_controllable
from flip_attack_optimizer.optimizer import FlipAttackOptimizer


def _make_run_start():
    traj = Trajectory()
    return RunStartEvent(trajectory=traj)


def _make_run_end():
    return RunEndEvent(evaluation=None)


def _make_pre_call(ctrl):
    return ControllablePreCallEvent(controllable=ctrl, request="user input")


def _make_post_call(ctrl, answer):
    return ControllablePostCallEvent(
        controllable=ctrl, request="user input", answer=answer
    )


async def _init_optimizer(**kwargs) -> FlipAttackOptimizer:
    defaults = dict(
        flip_modes=["FWO", "FCW"],
        use_cot=False,
        use_few_shot=False,
        success_threshold=0.8,
    )
    defaults.update(kwargs)
    opt = FlipAttackOptimizer(**defaults)
    mock_llm = AsyncMock()
    await opt.initialize(
        goal=Goal(description="test objective"),
        controllables=[make_controllable()],
        observables=[],
        llm_client=mock_llm,
    )
    return opt


@pytest.mark.asyncio
async def test_run_start_returns_event_response():
    opt = await _init_optimizer()
    resp = await opt.on_event(_make_run_start())
    assert isinstance(resp, EventResponse)


@pytest.mark.asyncio
async def test_pre_call_injects_flipped_task():
    opt = await _init_optimizer(flip_modes=["FWO"])
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableInjection)
    assert "TASK" in resp.value
    # FWO reverses word order: "test objective" -> "objective test"
    assert "objective test" in resp.value


@pytest.mark.asyncio
async def test_user_message_contains_system_prompt():
    """System prompt is folded into the user message."""
    opt = await _init_optimizer(flip_modes=["FWO"])
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableInjection)
    # Should contain system prompt content
    assert "Your primary mission is to solve the task" in resp.value
    # And the TASK line
    assert "TASK" in resp.value


@pytest.mark.asyncio
async def test_dual_controllable_flow_system_prompt_then_user_message():
    """ChatbotTarget-style flow: inject system prompt, then one user turn."""
    opt = await _init_optimizer(flip_modes=["FWO"])
    system_ctrl = make_controllable(name="system_prompt")
    user_ctrl = make_controllable(name="user_message")

    await opt.on_event(_make_run_start())

    sp_resp = await opt.on_event(_make_pre_call(system_ctrl))
    assert isinstance(sp_resp, ControllableInjection)
    assert "Step 1." in sp_resp.value

    user_resp = await opt.on_event(_make_pre_call(user_ctrl))
    assert isinstance(user_resp, ControllableInjection)
    assert "TASK is" in user_resp.value
    assert "Step 1." not in user_resp.value  # user payload should not fold system prompt

    # Single-turn optimizer: second user_message should stop.
    user_resp2 = await opt.on_event(_make_pre_call(user_ctrl))
    assert isinstance(user_resp2, ControllableNoInjection)


@pytest.mark.asyncio
async def test_dual_controllable_post_call_scored_on_response_channel():
    """PostCall can be accepted from response controllable after user_message injection."""
    opt = await _init_optimizer(flip_modes=["FWO"])
    system_ctrl = make_controllable(name="system_prompt")
    user_ctrl = make_controllable(name="user_message")
    response_ctrl = make_controllable(name="response")

    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(system_ctrl))
    injected = await opt.on_event(_make_pre_call(user_ctrl))
    assert isinstance(injected, ControllableInjection)

    opt.llm.complete.return_value = mock_response("Rating: [[8]]")
    post = await opt.on_event(
        ControllablePostCallEvent(
            controllable=response_ctrl,
            request=injected.value,
            answer="model answer",
        )
    )
    assert isinstance(post, ControllableNoInjection)
    assert opt._best_score == 0.0
    await opt.on_event(_make_run_end())
    assert opt._best_score > 0.0


@pytest.mark.asyncio
async def test_post_call_scores_and_returns_no_injection():
    opt = await _init_optimizer()
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    opt.llm.complete.return_value = mock_response("Rating: [[3]]")
    resp = await opt.on_event(_make_post_call(ctrl, "some response"))
    assert isinstance(resp, ControllableNoInjection)
    assert opt._best_score == 0.0
    await opt.on_event(_make_run_end())
    assert opt._best_score > 0.0


@pytest.mark.asyncio
async def test_success_triggers_done():
    opt = await _init_optimizer(success_threshold=0.8)
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    # Official ASR-GPT success is only rating == 10.
    opt.llm.complete.return_value = mock_response("Rating: [[9]]")
    await opt.on_event(_make_post_call(ctrl, "full compliance"))
    resp = await opt.on_event(_make_run_end())
    assert isinstance(resp, RunEndResponse)
    assert resp.done is False


@pytest.mark.asyncio
async def test_rating_10_triggers_done():
    opt = await _init_optimizer(success_threshold=0.8)
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    opt.llm.complete.return_value = mock_response("Rating: [[10]]")
    await opt.on_event(_make_post_call(ctrl, "full compliance"))
    resp = await opt.on_event(_make_run_end())
    assert isinstance(resp, RunEndResponse)
    assert resp.done is True


@pytest.mark.asyncio
async def test_scoring_exception_does_not_trigger_asr_gpt_success():
    opt = await _init_optimizer(flip_modes=["FWO"], success_threshold=0.8)
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    # Simulate stale prior state and then a scorer failure.
    assert opt._evaluator is not None
    opt._evaluator._last_rating = 10
    opt.llm.complete.side_effect = RuntimeError("judge failed")

    await opt.on_event(_make_post_call(ctrl, "some response"))
    resp = await opt.on_event(_make_run_end())
    assert isinstance(resp, RunEndResponse)
    assert opt._succeeded is False
    assert resp.done is True  # single strategy exhausted


@pytest.mark.asyncio
async def test_strategy_rotation():
    opt = await _init_optimizer(flip_modes=["FWO", "FCW"])
    ctrl = make_controllable()

    # First attempt (FWO)
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    opt.llm.complete.return_value = mock_response("Rating: [[2]]")
    await opt.on_event(_make_post_call(ctrl, "refused"))
    resp = await opt.on_event(_make_run_end())
    assert resp.done is False

    # Second attempt (FCW)
    await opt.on_event(_make_run_start())
    assert opt._attempt == 1
    resp2 = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp2, ControllableInjection)
    # FCW reverses chars in each word: "test" -> "tset", "objective" -> "evitcejbo"
    assert "tset" in resp2.value or "evitcejbo" in resp2.value


@pytest.mark.asyncio
async def test_all_strategies_exhausted():
    opt = await _init_optimizer(flip_modes=["FWO"])
    ctrl = make_controllable()

    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    opt.llm.complete.return_value = mock_response("Rating: [[1]]")
    await opt.on_event(_make_post_call(ctrl, "refused"))
    resp = await opt.on_event(_make_run_end())
    assert resp.done is True


@pytest.mark.asyncio
async def test_trajectory_response_preferred_over_postcall_answer():
    opt = await _init_optimizer(flip_modes=["FWO"])
    ctrl = make_controllable()
    run_start = _make_run_start()
    await opt.on_event(run_start)
    await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "post-call answer"))

    observable = Observable(name="response", security_domain=DOMAIN, description="response")
    run_start.trajectory.emit(
        ObservableEvent(observable=observable, content="trajectory answer")
    )

    opt._score_answer = AsyncMock()  # type: ignore[method-assign]
    await opt.on_event(_make_run_end())
    opt._score_answer.assert_awaited_once_with("trajectory answer")


@pytest.mark.asyncio
async def test_postcall_answer_used_when_no_trajectory_response():
    opt = await _init_optimizer(flip_modes=["FWO"])
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "post-call answer"))

    opt._score_answer = AsyncMock()  # type: ignore[method-assign]
    await opt.on_event(_make_run_end())
    opt._score_answer.assert_awaited_once_with("post-call answer")


@pytest.mark.asyncio
async def test_trajectory_response_heuristic_name_is_used():
    opt = await _init_optimizer(flip_modes=["FWO"])
    ctrl = make_controllable()
    run_start = _make_run_start()
    await opt.on_event(run_start)
    await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "post-call answer"))

    observable = Observable(
        name="assistant_output_text",
        security_domain=DOMAIN,
        description="response",
    )
    run_start.trajectory.emit(
        ObservableEvent(observable=observable, content="heuristic trajectory answer")
    )

    opt._score_answer = AsyncMock()  # type: ignore[method-assign]
    await opt.on_event(_make_run_end())
    opt._score_answer.assert_awaited_once_with("heuristic trajectory answer")


@pytest.mark.asyncio
async def test_custom_response_observable_names_are_used():
    opt = await _init_optimizer(
        flip_modes=["FWO"],
        response_observable_names=["target_reply_text"],
    )
    ctrl = make_controllable()
    run_start = _make_run_start()
    await opt.on_event(run_start)
    await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "post-call answer"))

    observable = Observable(
        name="target_reply_text",
        security_domain=DOMAIN,
        description="response",
    )
    run_start.trajectory.emit(
        ObservableEvent(observable=observable, content="custom trajectory answer")
    )

    opt._score_answer = AsyncMock()  # type: ignore[method-assign]
    await opt.on_event(_make_run_end())
    opt._score_answer.assert_awaited_once_with("custom trajectory answer")


@pytest.mark.asyncio
async def test_no_feedback_guardrail_stops_after_threshold():
    opt = await _init_optimizer(
        flip_modes=["FWO", "FCW"],
        max_no_feedback_runs=1,
    )
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))
    resp = await opt.on_event(_make_run_end())
    assert isinstance(resp, RunEndResponse)
    assert resp.done is True


@pytest.mark.asyncio
async def test_secondary_controllable_ignored():
    opt = await _init_optimizer()
    ctrl1 = make_controllable()
    ctrl2 = make_controllable(name="other")

    await opt.on_event(_make_run_start())
    resp1 = await opt.on_event(_make_pre_call(ctrl1))
    assert isinstance(resp1, ControllableInjection)

    resp2 = await opt.on_event(_make_pre_call(ctrl2))
    assert isinstance(resp2, ControllableNoInjection)


@pytest.mark.asyncio
async def test_system_prompt_exposed():
    opt = await _init_optimizer(flip_modes=["FWO"], prompt_style="vanilla")
    await opt.on_event(_make_run_start())
    assert opt.system_prompt is not None
    assert isinstance(opt.system_prompt, str)
    assert len(opt.system_prompt) > 50
