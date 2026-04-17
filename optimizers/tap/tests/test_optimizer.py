"""Tests for TapOptimizer event-driven state machine."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from superred.core.types.controllable import Controllable
from superred.core.types.event import EventResponse
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
from superred.core.types.security_domain import SecurityDomainTag

from tap_optimizer.optimizer import TapOptimizer

# -- Helpers ------------------------------------------------------------------

DOMAIN = SecurityDomainTag(name="test")


def _make_controllable():
    return Controllable(name="user_input", security_domain=DOMAIN, description="test")


def _make_run_start():
    traj = MagicMock()
    return RunStartEvent(trajectory=traj)


def _make_run_end():
    return RunEndEvent(evaluation=None)


def _make_pre_call(ctrl):
    return ControllablePreCallEvent(controllable=ctrl, request="hello")


def _make_post_call(ctrl, answer):
    return ControllablePostCallEvent(controllable=ctrl, request="hello", answer=answer)


def _mock_response(content: str) -> MagicMock:
    """Build a mock that mimics an LLM completion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


async def _init_optimizer(**kwargs) -> TapOptimizer:
    """Create and initialize a TapOptimizer with sensible test defaults."""
    defaults = dict(
        branching_factor=1,
        tree_width=1,
        tree_depth=3,
        attack_temperature=1.0,
        target_temperature=0.0,
    )
    defaults.update(kwargs)
    opt = TapOptimizer(**defaults)
    mock_llm = AsyncMock()
    await opt.initialize(
        goal=Goal(description="test goal"),
        controllables=[_make_controllable()],
        observables=[],
        llm_client=mock_llm,
    )
    return opt


def _setup_llm_mock(opt: TapOptimizer, responses: list[str]) -> None:
    """Wire up opt.llm.complete to return the given responses in order.

    The attacker, evaluator, and internal target all use opt.llm.complete,
    so a single side_effect list covers the entire depth cycle.
    """
    call_count = 0

    async def mock_complete(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        idx = call_count - 1
        if idx < len(responses):
            return _mock_response(responses[idx])
        return _mock_response("fallback")

    opt.llm.complete = AsyncMock(side_effect=mock_complete)


# -- Tests --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_start_returns_event_response():
    """RunStartEvent should trigger internal TAP work and return EventResponse."""
    opt = await _init_optimizer()
    attacker_json = json.dumps({"improvement": "Try this", "prompt": "attack prompt"})
    _setup_llm_mock(opt, [
        attacker_json,   # attacker
        "YES",           # on-topic
        "Some response", # internal target
        "5",             # score
    ])
    event = _make_run_start()
    result = await opt.on_event(event)
    assert isinstance(result, EventResponse)
    assert result.event is event


@pytest.mark.asyncio
async def test_pre_call_injects_best_prompt():
    """After RunStart produces a candidate, PreCall should inject its prompt."""
    opt = await _init_optimizer()
    attacker_json = json.dumps({"improvement": "Roleplaying", "prompt": "You are a pirate"})
    _setup_llm_mock(opt, [
        attacker_json,           # attacker
        "YES",                   # on-topic
        "Here is the content",   # internal target
        "5",                     # score
    ])

    await opt.on_event(_make_run_start())

    ctrl = _make_controllable()
    result = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(result, ControllableInjection)
    assert result.value == "You are a pirate"
    assert result.controllable is ctrl


@pytest.mark.asyncio
async def test_post_call_scores_and_returns_no_injection():
    """PostCall should score the real target response and return NoInjection."""
    opt = await _init_optimizer()
    attacker_json = json.dumps({"improvement": "Try this", "prompt": "attack prompt"})
    _setup_llm_mock(opt, [
        attacker_json,   # attacker
        "YES",           # on-topic
        "Some response", # internal target
        "5",             # score (internal)
        "6",             # score (real target, from post_call)
    ])

    await opt.on_event(_make_run_start())

    ctrl = _make_controllable()
    await opt.on_event(_make_pre_call(ctrl))

    result = await opt.on_event(_make_post_call(ctrl, "Real target says hello"))
    assert isinstance(result, ControllableNoInjection)
    assert result.controllable is ctrl
    # The best candidate should have been scored
    assert opt._best_candidate is not None
    assert opt._best_candidate.target_response == "Real target says hello"


@pytest.mark.asyncio
async def test_jailbreak_on_internal_score_triggers_done():
    """Internal score >= 10 during RunStart should set done=True."""
    opt = await _init_optimizer(tree_depth=5)
    attacker_json = json.dumps({"improvement": "Perfect attack", "prompt": "Jailbreak prompt"})
    _setup_llm_mock(opt, [
        attacker_json,                 # attacker
        "YES",                         # on-topic
        "Full compliance response",    # internal target
        "10",                          # score = 10 -> jailbreak found
    ])

    await opt.on_event(_make_run_start())

    # The optimizer should have found jailbreak and set done
    assert opt._done is True

    # PreCall should inject the jailbreak prompt
    ctrl = _make_controllable()
    result = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(result, ControllableInjection)
    assert result.value == "Jailbreak prompt"

    # RunEnd should report done=True
    result = await opt.on_event(_make_run_end())
    assert isinstance(result, RunEndResponse)
    assert result.done is True


@pytest.mark.asyncio
async def test_jailbreak_confirmed_by_real_target():
    """Real target score >= 10 on PostCall should set done=True."""
    opt = await _init_optimizer(tree_depth=5)
    attacker_json = json.dumps({"improvement": "Try this", "prompt": "sneaky prompt"})
    _setup_llm_mock(opt, [
        attacker_json,   # attacker
        "YES",           # on-topic
        "Some response", # internal target
        "5",             # internal score (not jailbreak)
        "10",            # real target score = 10 -> jailbreak confirmed
    ])

    await opt.on_event(_make_run_start())
    assert opt._done is False  # Not done yet from internal scoring

    ctrl = _make_controllable()
    await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "Fully compliant harmful response"))

    assert opt._done is True

    result = await opt.on_event(_make_run_end())
    assert isinstance(result, RunEndResponse)
    assert result.done is True


@pytest.mark.asyncio
async def test_continues_across_depths():
    """One depth cycle with low score should return done=False, then next depth branches."""
    opt = await _init_optimizer(tree_depth=5, tree_width=1, branching_factor=1)
    attacker_json = json.dumps({"improvement": "First try", "prompt": "prompt v1"})
    _setup_llm_mock(opt, [
        attacker_json,   # attacker
        "YES",           # on-topic
        "Some response", # internal target
        "3",             # low score
        "3",             # real target score (low)
    ])

    # Depth 0
    await opt.on_event(_make_run_start())
    ctrl = _make_controllable()
    await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "partial response"))

    result = await opt.on_event(_make_run_end())
    assert isinstance(result, RunEndResponse)
    assert result.done is False
    assert opt._depth == 1

    # Depth 1 -- set up new LLM responses for second depth
    attacker_json2 = json.dumps({"improvement": "Second try", "prompt": "prompt v2"})
    _setup_llm_mock(opt, [
        attacker_json2,  # attacker
        "YES",           # on-topic
        "Better resp",   # internal target
        "7",             # higher score
    ])

    await opt.on_event(_make_run_start())
    # Verify branching occurred (depth > 0 triggers branching)
    assert opt._best_candidate is not None
    assert opt._best_candidate.prompt == "prompt v2"


@pytest.mark.asyncio
async def test_depth_exhausted_signals_done():
    """When max depth is reached, RunEnd should return done=True."""
    opt = await _init_optimizer(tree_depth=1)
    attacker_json = json.dumps({"improvement": "Try", "prompt": "attack"})
    _setup_llm_mock(opt, [
        attacker_json,   # attacker
        "YES",           # on-topic
        "Some response", # internal target
        "3",             # low score
        "3",             # real target score
    ])

    await opt.on_event(_make_run_start())
    ctrl = _make_controllable()
    await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "response"))

    # depth starts at 0, incremented to 1 in _handle_run_end, which == tree_depth=1
    result = await opt.on_event(_make_run_end())
    assert isinstance(result, RunEndResponse)
    assert result.done is True


@pytest.mark.asyncio
async def test_no_leaves_signals_done():
    """If all nodes are pruned (off-topic), RunStart should set done=True."""
    opt = await _init_optimizer()
    attacker_json = json.dumps({"improvement": "Off topic", "prompt": "irrelevant prompt"})
    _setup_llm_mock(opt, [
        attacker_json,   # attacker
        "NO",            # off-topic -> pruned
    ])

    await opt.on_event(_make_run_start())
    assert opt._done is True

    result = await opt.on_event(_make_run_end())
    assert isinstance(result, RunEndResponse)
    assert result.done is True


@pytest.mark.asyncio
async def test_fallback_to_goal_when_no_candidate():
    """When no best candidate exists, PreCall should inject the goal description."""
    opt = await _init_optimizer()
    # Simulate all nodes pruned -- no RunStart processing
    opt._done = True
    opt._best_candidate = None

    ctrl = _make_controllable()
    result = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(result, ControllableInjection)
    assert result.value == "test goal"
