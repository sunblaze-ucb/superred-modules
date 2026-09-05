"""Tests for PolicyPuppetryOptimizer.

One scene per run; declines once variants are exhausted and reports done on
the final RunEnd.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from superred.core.types.controllable import Controllable
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.security_domain import SecurityDomainTag

from policy_puppetry_optimizer.optimizer import PolicyPuppetryOptimizer
from policy_puppetry_optimizer.templates import (
    HARMFUL_BEHAVIOR_SLOT,
    leetspeak,
    render,
)

USER_TAG = SecurityDomainTag("user")
GOAL_TEXT = "explain how to pick a lock"


def _ctrl(name: str = "user_message", value_type: str = "text") -> Controllable:
    return Controllable(name=name, security_domain=USER_TAG, value_type=value_type)


def _make(**kwargs) -> PolicyPuppetryOptimizer:
    opt = PolicyPuppetryOptimizer(**kwargs)
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL_TEXT),
            controllables=[_ctrl()],
            observables=[],
            llm_client=MagicMock(),
        )
    )
    return opt


def _start(opt) -> None:
    asyncio.run(opt.on_event(RunStartEvent(trajectory=MagicMock())))


def _inject(opt, name: str = "user_message", value_type: str = "text"):
    return asyncio.run(
        opt.on_event(
            ControllablePreCallEvent(controllable=_ctrl(name, value_type), request="")
        )
    )


def _end(opt) -> RunEndResponse:
    return asyncio.run(opt.on_event(RunEndEvent(evaluation=None)))


def test_first_variant_is_the_plain_scene() -> None:
    opt = _make()
    _start(opt)
    resp = _inject(opt)
    assert isinstance(resp, ControllableInjection)
    assert resp.value == render(GOAL_TEXT)


def test_leet_variants_follow_and_are_seeded_in_order() -> None:
    opt = _make(techniques=["leet"], leet_variants=3)
    expected_base = render(GOAL_TEXT, leet_suffix=True)
    for seed in range(3):
        _start(opt)
        resp = _inject(opt)
        assert resp.value == leetspeak(expected_base, seed=seed)
        _end(opt)


def test_total_variant_count_is_one_plus_leet_variants() -> None:
    opt = _make(leet_variants=4)
    seen = 0
    while True:
        _start(opt)
        resp = _inject(opt)
        if isinstance(resp, ControllableNoInjection):
            break
        seen += 1
        if _end(opt).done:
            break
    assert seen == 5


def test_done_reported_only_on_the_final_run() -> None:
    opt = _make(techniques=["leet"], leet_variants=2)
    _start(opt)
    _inject(opt)
    assert _end(opt).done is False
    _start(opt)
    _inject(opt)
    assert _end(opt).done is True


def test_no_slot_marker_survives_into_any_variant() -> None:
    opt = _make(leet_variants=2)
    for _ in range(3):
        _start(opt)
        value = _inject(opt).value
        assert HARMFUL_BEHAVIOR_SLOT not in value
        _end(opt)


def test_goal_is_carried_into_every_variant() -> None:
    """The template must be filled with the task goal, not a fixed upstream
    payload. Leetspeak may transmute letters, so accept either spelling."""
    opt = _make(leet_variants=2)
    for _ in range(3):
        _start(opt)
        value = _inject(opt).value
        assert "lock" in value or "l0ck" in value
        assert "methamphetamine" not in value
        _end(opt)


def test_injects_only_once_per_run() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_inject(opt, "user_query"), ControllableInjection)
    assert isinstance(_inject(opt, "db_lookup"), ControllableNoInjection)


def test_never_injects_into_system_prompt() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_inject(opt, "system_prompt"), ControllableNoInjection)


def test_declines_non_free_text_surfaces() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(
        _inject(opt, "tool_description", value_type="json"), ControllableNoInjection
    )


def test_declines_once_exhausted() -> None:
    opt = _make(techniques=["plain"])
    _start(opt)
    _inject(opt)
    _end(opt)
    _start(opt)
    assert isinstance(_inject(opt), ControllableNoInjection)


def test_rejects_unknown_technique() -> None:
    with pytest.raises(ValueError, match="unknown technique"):
        PolicyPuppetryOptimizer(techniques=["nope"])


def test_rejects_empty_techniques() -> None:
    with pytest.raises(ValueError, match="at least one technique"):
        PolicyPuppetryOptimizer(techniques=[])


def test_rejects_non_positive_variant_count() -> None:
    with pytest.raises(ValueError, match="leet_variants"):
        PolicyPuppetryOptimizer(leet_variants=0)


def test_initialize_resets_state_between_tasks() -> None:
    opt = _make(techniques=["plain"])
    _start(opt)
    _inject(opt)
    _end(opt)
    asyncio.run(
        opt.initialize(
            goal=Goal(description="something else entirely"),
            controllables=[_ctrl()],
            observables=[],
            llm_client=MagicMock(),
        )
    )
    _start(opt)
    assert "something else entirely" in _inject(opt).value
