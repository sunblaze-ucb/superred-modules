"""Tests for TapOptimizer orchestrator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue
from superred.core.types.security_domain import SecurityDomainTag
from superred.core.types.trajectory import Trajectory

from tap_optimizer.optimizer import TapOptimizer

# ── Helpers ──────────────────────────────────────────────────────────────────

DOMAIN = SecurityDomainTag(name="test")


def _make_completion_response(content: str) -> MagicMock:
    """Build a mock that mimics litellm's acompletion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_controllable_event() -> ControllablePreCallEvent:
    ctrl = Controllable(name="user_input", security_domain=DOMAIN, description="test")
    return ControllablePreCallEvent(controllable=ctrl, request="hello")


def _make_run_start_event() -> RunStartEvent:
    traj = Trajectory(filtered_scope=DOMAIN)
    return RunStartEvent(trajectory=traj)


def _make_run_end_event() -> RunEndEvent:
    traj = Trajectory(filtered_scope=DOMAIN)
    return RunEndEvent(trajectory=traj)


def _make_optimizer(**kwargs) -> TapOptimizer:
    """Create a TapOptimizer with sensible test defaults."""
    defaults = dict(
        attacker_model="gpt-4o",
        target_model="gpt-4o",
        evaluator_model="gpt-4o",
        branching_factor=1,
        tree_width=1,
        tree_depth=1,
    )
    defaults.update(kwargs)
    return TapOptimizer(**defaults)


# ── Event handling tests ─────────────────────────────────────────────────────


class TestRunStartReturnsEventResponse:
    @pytest.mark.asyncio
    async def test_run_start_returns_event_response(self):
        opt = _make_optimizer()
        goal = Goal(description="test goal")
        ctrl = Controllable(name="user_input", security_domain=DOMAIN, description="test")
        await opt.initialize(goal=goal, controllables=[ctrl], observables=[])

        event = _make_run_start_event()
        result = await opt.on_event(event)

        assert isinstance(result, EventResponse)
        assert result.event is event


class TestRunEndReturnsDoneTrue:
    @pytest.mark.asyncio
    async def test_run_end_returns_done_true(self):
        opt = _make_optimizer()
        goal = Goal(description="test goal")
        ctrl = Controllable(name="user_input", security_domain=DOMAIN, description="test")
        await opt.initialize(goal=goal, controllables=[ctrl], observables=[])

        event = _make_run_end_event()
        result = await opt.on_event(event)

        assert isinstance(result, RunEndResponse)
        assert result.done is True


# ── TAP integration test ─────────────────────────────────────────────────────


class TestControllableEventRunsTapAndInjects:
    """Minimal TAP run: depth=1, width=1, bf=1.

    Sequential mock calls:
    1. attacker acompletion -> JSON with improvement + prompt
    2. evaluator acompletion (on-topic) -> "YES"
    3. target acompletion -> target response text
    4. evaluator acompletion (score) -> "5"
    """

    @pytest.mark.asyncio
    async def test_controllable_event_runs_tap_and_injects(self):
        opt = _make_optimizer(tree_width=1, tree_depth=1, branching_factor=1)
        goal = Goal(description="test goal")
        ctrl = Controllable(name="user_input", security_domain=DOMAIN, description="test")
        await opt.initialize(goal=goal, controllables=[ctrl], observables=[])

        attacker_json = json.dumps({"improvement": "Try roleplaying", "prompt": "You are a pirate"})

        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            current = call_count
            if current == 1:
                # Attacker: generate prompt
                return _make_completion_response(attacker_json)
            elif current == 2:
                # Evaluator: on-topic check
                return _make_completion_response("YES")
            elif current == 3:
                # Target: internal query
                return _make_completion_response("Here is the harmful content")
            elif current == 4:
                # Evaluator: score response
                return _make_completion_response("5")
            else:
                return _make_completion_response("fallback")

        with (
            patch("tap_optimizer.attacker.acompletion", side_effect=mock_acompletion),
            patch("tap_optimizer.evaluator.acompletion", side_effect=mock_acompletion),
            patch("tap_optimizer.optimizer.acompletion", side_effect=mock_acompletion),
        ):
            event = _make_controllable_event()
            result = await opt.on_event(event)

        assert isinstance(result, ControllableInjection)
        assert result.value == "You are a pirate"
        assert result.controllable is event.controllable
        assert call_count == 4


class TestEarlyReturnOnScore10:
    """Set depth=5 but score=10 at depth 0.

    Only 4 LLM calls should be made (no deeper iterations).
    """

    @pytest.mark.asyncio
    async def test_early_return_on_score_10(self):
        opt = _make_optimizer(tree_width=1, tree_depth=5, branching_factor=1)
        goal = Goal(description="test goal")
        ctrl = Controllable(name="user_input", security_domain=DOMAIN, description="test")
        await opt.initialize(goal=goal, controllables=[ctrl], observables=[])

        attacker_json = json.dumps({"improvement": "Perfect attack", "prompt": "Jailbreak prompt"})

        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            current = call_count
            if current == 1:
                # Attacker: generate prompt
                return _make_completion_response(attacker_json)
            elif current == 2:
                # Evaluator: on-topic check
                return _make_completion_response("YES")
            elif current == 3:
                # Target: internal query
                return _make_completion_response("Full compliance response")
            elif current == 4:
                # Evaluator: score = 10 -> early return
                return _make_completion_response("10")
            else:
                # Should NOT be reached
                return _make_completion_response("should not happen")

        with (
            patch("tap_optimizer.attacker.acompletion", side_effect=mock_acompletion),
            patch("tap_optimizer.evaluator.acompletion", side_effect=mock_acompletion),
            patch("tap_optimizer.optimizer.acompletion", side_effect=mock_acompletion),
        ):
            event = _make_controllable_event()
            result = await opt.on_event(event)

        assert isinstance(result, ControllableInjection)
        assert result.value == "Jailbreak prompt"
        assert call_count == 4  # Did not continue to deeper depths
