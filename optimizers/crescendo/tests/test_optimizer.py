"""Tests for the CrescendoOptimizer event-driven state machine."""

from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    ControllablePostCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue
from superred.core.types.security_domain import SecurityDomainTag

from crescendo_optimizer.optimizer import CrescendoOptimizer


def _make_controllable(name: str = "input", tag: str = "user"):
    return Controllable(name=name, security_domain=SecurityDomainTag(name=tag))


def _make_run_start():
    traj = MagicMock()
    return RunStartEvent(trajectory=traj)


def _make_run_end():
    return RunEndEvent(evaluation=None)


def _make_pre_call(ctrl):
    return ControllablePreCallEvent(controllable=ctrl, request="user input")


def _make_post_call(ctrl, answer):
    return ControllablePostCallEvent(controllable=ctrl, request="user input", answer=answer)


async def _init_optimizer(**kwargs) -> CrescendoOptimizer:
    defaults = dict(
        max_turns=3,
        max_backtracks=2,
        success_threshold=0.8,
        max_attempts=2,
    )
    defaults.update(kwargs)
    opt = CrescendoOptimizer(**defaults)

    mock_llm = AsyncMock()
    await opt.initialize(
        goal=Goal(description="test objective"),
        controllables=[_make_controllable()],
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
async def test_pre_call_injects_question():
    """ControllablePreCallEvent should trigger attacker and inject a question."""
    opt = await _init_optimizer()
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("What is X?", "", "Starting innocuous")
        resp = await opt.on_event(_make_pre_call(ctrl))

    assert isinstance(resp, ControllableInjection)
    assert resp.value == "What is X?"


@pytest.mark.asyncio
async def test_post_call_evaluates_and_returns_no_injection():
    """ControllablePostCallEvent should evaluate the response."""
    opt = await _init_optimizer()
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q?", "", "reason")
        await opt.on_event(_make_pre_call(ctrl))

    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.3, "Low progress")
        resp = await opt.on_event(_make_post_call(ctrl, "Here is some info about X"))

    assert isinstance(resp, ControllableNoInjection)


@pytest.mark.asyncio
async def test_post_call_with_different_controllable_is_processed():
    """PostCall should still be processed when target uses a separate response controllable."""
    opt = await _init_optimizer()
    pre_ctrl = _make_controllable(name="user_message", tag="user")
    post_ctrl = _make_controllable(name="response", tag="assistant")

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q?", "", "reason")
        pre_resp = await opt.on_event(_make_pre_call(pre_ctrl))
    assert isinstance(pre_resp, ControllableInjection)

    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.4, "Progress")
        post_resp = await opt.on_event(_make_post_call(post_ctrl, "Answer text"))

    assert isinstance(post_resp, ControllableNoInjection)
    assert opt._turn == 1
    assert opt._last_response == "Answer text"


@pytest.mark.asyncio
async def test_multi_turn_within_single_run():
    """Multiple PreCall/PostCall cycles happen within one run."""
    opt = await _init_optimizer(max_turns=3, success_threshold=0.99)
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    for turn in range(3):
        with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = (f"Q{turn + 1}?", "", "r")
            resp = await opt.on_event(_make_pre_call(ctrl))
        assert isinstance(resp, ControllableInjection)
        assert resp.value == f"Q{turn + 1}?"

        with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
             patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
            mock_ref.return_value = False
            mock_score.return_value = (0.2 * (turn + 1), "Progress")
            await opt.on_event(_make_post_call(ctrl, f"Answer {turn + 1}"))

    assert opt._turn == 3

    # After max_turns, next PreCall should get NoInjection
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableNoInjection)


@pytest.mark.asyncio
async def test_success_mid_conversation_stops_injection():
    """Score >= threshold should stop injecting on next PreCall."""
    opt = await _init_optimizer(max_turns=5, success_threshold=0.8)
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    # Turn 1: low score
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q1?", "", "r")
        await opt.on_event(_make_pre_call(ctrl))
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.3, "Low")
        await opt.on_event(_make_post_call(ctrl, "partial"))

    # Turn 2: high score — success
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q2?", "", "r")
        await opt.on_event(_make_pre_call(ctrl))
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.9, "High")
        await opt.on_event(_make_post_call(ctrl, "harmful"))

    assert opt._succeeded is True

    # Next PreCall should get NoInjection
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableNoInjection)

    # RunEnd should signal done
    resp = await opt.on_event(_make_run_end())
    assert resp.done is True


@pytest.mark.asyncio
async def test_backtracking_on_refusal():
    """Refusal should trigger backtracking without incrementing turn."""
    opt = await _init_optimizer()
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Direct harmful Q?", "", "r")
        await opt.on_event(_make_pre_call(ctrl))

    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._attacker, "notify_refusal") as mock_notify:
        mock_ref.return_value = True
        await opt.on_event(_make_post_call(ctrl, "I can't help with that"))
        mock_notify.assert_called_once_with("Direct harmful Q?")

    # Turn should not have incremented, attempt not done
    assert opt._turn == 0
    assert opt._attempt_done is False


@pytest.mark.asyncio
async def test_max_backtracks_stops_backtracking():
    """After max_backtracks, refusals are treated as normal responses."""
    opt = await _init_optimizer(max_backtracks=1, max_turns=5)
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    # First refusal — should backtrack
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q1?", "", "r")
        await opt.on_event(_make_pre_call(ctrl))
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._attacker, "notify_refusal"):
        mock_ref.return_value = True
        await opt.on_event(_make_post_call(ctrl, "I can't"))
    assert opt._backtrack_count == 1
    assert opt._turn == 0

    # Second refusal — backtracking exhausted, treat as normal response
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q2?", "", "r")
        await opt.on_event(_make_pre_call(ctrl))
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = True
        mock_score.return_value = (0.0, "Refusal")
        await opt.on_event(_make_post_call(ctrl, "I can't again"))

    assert opt._turn == 1
    assert opt._backtrack_count == 1  # didn't increase


@pytest.mark.asyncio
async def test_variant_rotation_across_runs():
    """Each run uses a different variant. Failed attempt → next variant."""
    opt = await _init_optimizer(max_turns=1, max_attempts=2)
    ctrl = _make_controllable()

    # Run 1 (attempt 0): exhaust max_turns
    await opt.on_event(_make_run_start())
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q?", "", "r")
        await opt.on_event(_make_pre_call(ctrl))
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.2, "Low")
        await opt.on_event(_make_post_call(ctrl, "response"))
    # NoInjection to terminate target loop
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableNoInjection)

    resp = await opt.on_event(_make_run_end())
    assert resp.done is False  # more attempts available

    # Run 2 (attempt 1): should reset state
    await opt.on_event(_make_run_start())
    assert opt._attempt == 1
    assert opt._turn == 0


@pytest.mark.asyncio
async def test_all_attempts_exhausted():
    """When all attempts exhausted, RunEnd signals done=True."""
    opt = await _init_optimizer(max_turns=1, max_attempts=1)
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q?", "", "r")
        await opt.on_event(_make_pre_call(ctrl))
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.3, "Low")
        await opt.on_event(_make_post_call(ctrl, "response"))

    # Terminate target loop
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableNoInjection)

    resp = await opt.on_event(_make_run_end())
    assert resp.done is True


@pytest.mark.asyncio
async def test_secondary_controllable_ignored():
    """Non-primary controllable should get NoInjection."""
    opt = await _init_optimizer()
    ctrl1 = _make_controllable()
    ctrl2 = Controllable(name="other", security_domain=SecurityDomainTag(name="sys"), description="secondary")

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q?", "", "r")
        resp1 = await opt.on_event(_make_pre_call(ctrl1))
    assert isinstance(resp1, ControllableInjection)

    resp2 = await opt.on_event(_make_pre_call(ctrl2))
    assert isinstance(resp2, ControllableNoInjection)


@pytest.mark.asyncio
async def test_attacker_failure_uses_benign_fallback_question():
    opt = await _init_optimizer()
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = RuntimeError("attacker unavailable")
        resp = await opt.on_event(_make_pre_call(ctrl))

    assert isinstance(resp, ControllableInjection)
    assert "test objective" not in resp.value.lower()
    assert "high-level" in resp.value.lower()
