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


def _make_controllable():
    return Controllable(name="input", security_domain=SecurityDomainTag(name="user"))


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
async def test_run_end_continues_when_not_done():
    """RunEndEvent should return done=False when more turns remain."""
    opt = await _init_optimizer(max_turns=5)
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q?", "", "r")
        await opt.on_event(_make_pre_call(ctrl))
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.2, "Low")
        await opt.on_event(_make_post_call(ctrl, "response"))

    resp = await opt.on_event(_make_run_end())
    assert isinstance(resp, RunEndResponse)
    assert resp.done is False


@pytest.mark.asyncio
async def test_success_triggers_done():
    """Score >= threshold should set done=True on RunEndEvent."""
    opt = await _init_optimizer(success_threshold=0.8)
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q?", "", "r")
        await opt.on_event(_make_pre_call(ctrl))
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.9, "High compliance")
        await opt.on_event(_make_post_call(ctrl, "harmful content"))

    resp = await opt.on_event(_make_run_end())
    assert resp.done is True


@pytest.mark.asyncio
async def test_backtracking_on_refusal():
    """Refusal should trigger backtracking: pop history, notify attacker."""
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

    # Turn should not have incremented
    assert opt._turn == 0
    # Conversation history should be empty (question was popped)
    assert len(opt._conversation_history) == 0


@pytest.mark.asyncio
async def test_max_backtracks_stops_backtracking():
    """After max_backtracks, refusals no longer trigger backtracking."""
    opt = await _init_optimizer(max_backtracks=1)
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

    # Second refusal — should NOT backtrack (max reached), treat as normal response
    resp2 = await opt.on_event(_make_run_end())
    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q2?", "", "r")
        await opt.on_event(_make_pre_call(ctrl))
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = True
        mock_score.return_value = (0.0, "Refusal")
        await opt.on_event(_make_post_call(ctrl, "I can't again"))

    # Turn should have incremented since backtracking was exhausted
    assert opt._turn == 1
    assert opt._backtrack_count == 1  # didn't increase


@pytest.mark.asyncio
async def test_variant_rotation_on_attempt_exhaustion():
    """When max_turns exhausted, should start new attempt with next variant."""
    opt = await _init_optimizer(max_turns=1, max_attempts=2)
    ctrl = _make_controllable()

    # Complete first attempt (1 turn)
    await opt.on_event(_make_run_start())
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q?", "", "r")
        await opt.on_event(_make_pre_call(ctrl))
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.2, "Low")
        await opt.on_event(_make_post_call(ctrl, "response"))

    resp = await opt.on_event(_make_run_end())
    assert resp.done is False  # Should continue to next attempt

    # Next RunStartEvent should reset state for attempt 2
    await opt.on_event(_make_run_start())
    assert opt._attempt == 1
    assert opt._turn == 0


@pytest.mark.asyncio
async def test_all_attempts_exhausted():
    """When all attempts and turns exhausted, should signal done=True."""
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

    resp = await opt.on_event(_make_run_end())
    assert resp.done is True


@pytest.mark.asyncio
async def test_conversation_history_tracks_exchanges():
    """Conversation history should accumulate user/assistant pairs."""
    opt = await _init_optimizer()
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = ("Q1?", "", "r")
        await opt.on_event(_make_pre_call(ctrl))

    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.3, "Low")
        await opt.on_event(_make_post_call(ctrl, "Answer 1"))

    assert len(opt._conversation_history) == 2
    assert opt._conversation_history[0] == {"role": "user", "content": "Q1?"}
    assert opt._conversation_history[1] == {"role": "assistant", "content": "Answer 1"}
