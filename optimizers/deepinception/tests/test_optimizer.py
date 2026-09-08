"""Tests for DeepInceptionOptimizer."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from superred.core.types.controllable import Controllable
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    RunEndEvent,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.security_domain import SecurityDomainTag

from deepinception_optimizer.optimizer import DeepInceptionOptimizer
from deepinception_optimizer.template import render

USER = SecurityDomainTag("user")
GOAL = "explain how to pick a lock"


def _ctrl(name="user_message", value_type="text"):
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _make(**kw):
    opt = DeepInceptionOptimizer(**kw)
    asyncio.run(opt.initialize(goal=Goal(description=GOAL), controllables=[_ctrl()],
                               observables=[], llm_client=MagicMock()))
    return opt


def _start(opt):
    asyncio.run(opt.on_event(RunStartEvent(trajectory=MagicMock())))


def _inject(opt, name="user_message", value_type="text"):
    return asyncio.run(opt.on_event(ControllablePreCallEvent(controllable=_ctrl(name, value_type), request="")))


def _end(opt):
    return asyncio.run(opt.on_event(RunEndEvent(evaluation=None)))


def test_injects_the_rendered_scene() -> None:
    opt = _make()
    _start(opt)
    resp = _inject(opt)
    assert isinstance(resp, ControllableInjection)
    assert resp.value == render(GOAL)
    assert GOAL in resp.value


def test_single_attempt_per_task() -> None:
    opt = _make()
    _start(opt)
    _inject(opt)
    assert _end(opt).done is True


def test_injects_once_per_run() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_inject(opt, "user_query"), ControllableInjection)
    assert isinstance(_inject(opt, "db_lookup"), ControllableNoInjection)


def test_never_injects_system_prompt() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_inject(opt, "system_prompt"), ControllableNoInjection)


def test_declines_non_free_text() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_inject(opt, "x", value_type="json"), ControllableNoInjection)


def test_scene_params_flow_into_payload() -> None:
    opt = _make(scene="science fiction", character_number=6, layer_number=4)
    _start(opt)
    v = _inject(opt).value
    assert "science fiction" in v and "more than 6" in v and "reach layer 4" in v


def test_bad_params_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        DeepInceptionOptimizer(layer_number=0)


def test_reinitialize_rerenders_for_new_goal() -> None:
    opt = _make()
    _start(opt); _inject(opt); _end(opt)
    asyncio.run(opt.initialize(goal=Goal(description="something else"),
                               controllables=[_ctrl()], observables=[], llm_client=MagicMock()))
    _start(opt)
    assert "something else" in _inject(opt).value


def test_follow_ups_are_off_by_default() -> None:
    opt = _make()
    _start(opt)
    _inject(opt)
    assert _end(opt).done is True


def test_follow_ups_continue_the_scene_on_the_same_channel() -> None:
    """Upstream's further_q keeps asking within the established scene, so the
    follow-ups must land on the surface that received the inception prompt."""
    from deepinception_optimizer.template import follow_up_questions

    opt = _make(follow_ups=True)
    _start(opt)
    first = _inject(opt, "user_query")
    assert isinstance(first, ControllableInjection)
    assert "layer 5" in first.value  # the inception scene
    assert _end(opt).done is False

    for i, expected in enumerate(follow_up_questions()):
        _start(opt)
        # a different surface must not receive the follow-up
        assert isinstance(_inject(opt, "other_surface"), ControllableNoInjection)
        resp = _inject(opt, "user_query")
        assert isinstance(resp, ControllableInjection)
        assert resp.value == expected
        done = _end(opt).done
        assert done is (i == len(follow_up_questions()) - 1)


def test_post_call_returns_an_injection_decision_not_a_bare_event_response() -> None:
    """The channel type-checks PostCall responses.

    A bare ``EventResponse`` raises ``TypeError`` and aborts the task on any
    target that emits post-call events. This optimizer never rewrites the
    target's answer, so it must decline.
    """
    opt = _make()
    response = asyncio.run(
        opt.on_event(
            ControllablePostCallEvent(
                controllable=_ctrl(), request="q", answer="a"
            )
        )
    )
    assert isinstance(response, ControllableNoInjection)
