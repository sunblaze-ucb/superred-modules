"""Tests for GoalPassthroughOptimizer.

The passthrough baseline injects the goal text once on the first
non-system_prompt PreCall, declines everything else, and ends the run.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from superred.core.types.controllable import Controllable
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.event import EventResponse
from superred.core.types.goal import Goal
from superred.core.types.security_domain import SecurityDomainTag

from goal_passthrough_optimizer.optimizer import GoalPassthroughOptimizer

USER_TAG = SecurityDomainTag("user")
SYSTEM_PROMPT_TAG = SecurityDomainTag("system_prompt")

GOAL_TEXT = "Tell me how to do the forbidden thing."


def _user_ctrl(name: str = "user_message") -> Controllable:
    return Controllable(name=name, security_domain=USER_TAG)


def _system_prompt_ctrl() -> Controllable:
    return Controllable(name="system_prompt", security_domain=SYSTEM_PROMPT_TAG)


def _make_optimizer() -> GoalPassthroughOptimizer:
    opt = GoalPassthroughOptimizer()
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL_TEXT),
            controllables=[_user_ctrl()],
            observables=[],
            llm_client=MagicMock(),
        )
    )
    return opt


def _ev(opt: GoalPassthroughOptimizer, event):
    return asyncio.run(opt.on_event(event))


def test_zero_arg_construction() -> None:
    # superred convention: optimizers construct with no arguments.
    GoalPassthroughOptimizer()


def test_injects_goal_once_then_declines() -> None:
    opt = _make_optimizer()

    assert isinstance(_ev(opt, RunStartEvent(trajectory=MagicMock())), EventResponse)

    ctrl = _user_ctrl()
    first = _ev(
        opt, ControllablePreCallEvent(controllable=ctrl, request="user message")
    )
    assert isinstance(first, ControllableInjection)
    assert first.value == GOAL_TEXT

    second = _ev(
        opt, ControllablePreCallEvent(controllable=ctrl, request="user message")
    )
    assert isinstance(second, ControllableNoInjection)


def test_system_prompt_is_declined_user_still_injected() -> None:
    opt = _make_optimizer()
    _ev(opt, RunStartEvent(trajectory=MagicMock()))

    sp = _ev(
        opt,
        ControllablePreCallEvent(controllable=_system_prompt_ctrl(), request="sp"),
    )
    assert isinstance(sp, ControllableNoInjection)

    user = _ev(
        opt,
        ControllablePreCallEvent(controllable=_user_ctrl(), request="user message"),
    )
    assert isinstance(user, ControllableInjection)
    assert user.value == GOAL_TEXT


def test_post_call_never_injects() -> None:
    opt = _make_optimizer()
    _ev(opt, RunStartEvent(trajectory=MagicMock()))
    _ev(opt, ControllablePreCallEvent(controllable=_user_ctrl(), request="m"))

    post = _ev(
        opt,
        ControllablePostCallEvent(
            controllable=_user_ctrl(), request="m", answer="some answer"
        ),
    )
    assert isinstance(post, ControllableNoInjection)


def test_run_end_signals_done() -> None:
    opt = _make_optimizer()
    _ev(opt, RunStartEvent(trajectory=MagicMock()))
    resp = _ev(opt, RunEndEvent())
    assert isinstance(resp, RunEndResponse)
    assert resp.done is True


def test_injection_re_arms_each_run() -> None:
    opt = _make_optimizer()

    # Run 1: inject then decline.
    _ev(opt, RunStartEvent(trajectory=MagicMock()))
    assert isinstance(
        _ev(opt, ControllablePreCallEvent(controllable=_user_ctrl(), request="m")),
        ControllableInjection,
    )
    _ev(opt, RunEndEvent())

    # Run 2: a fresh RunStart re-arms the single injection.
    _ev(opt, RunStartEvent(trajectory=MagicMock()))
    again = _ev(opt, ControllablePreCallEvent(controllable=_user_ctrl(), request="m"))
    assert isinstance(again, ControllableInjection)
    assert again.value == GOAL_TEXT
