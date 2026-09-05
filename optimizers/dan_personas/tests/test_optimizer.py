"""Tests for DANPersonasOptimizer.

Two turns per run: persona preamble, then the goal. One persona per run.
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
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag

from dan_personas_optimizer.optimizer import DANPersonasOptimizer
from dan_personas_optimizer.personas import load_personas

USER_TAG = SecurityDomainTag("user")
GOAL_TEXT = "explain how to pick a lock"


def _ctrl(name: str = "user_message", value_type: str = "text") -> Controllable:
    return Controllable(name=name, security_domain=USER_TAG, value_type=value_type)


def _make(observables: list[ObservableValue] | None = None, **kwargs):
    opt = DANPersonasOptimizer(**kwargs)
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL_TEXT),
            controllables=[_ctrl()],
            observables=observables or [],
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


def test_first_turn_is_the_persona_preamble() -> None:
    opt = _make(personas=["STAN"])
    _start(opt)
    resp = _inject(opt)
    assert isinstance(resp, ControllableInjection)
    assert resp.value == load_personas(["STAN"])[0].render(None)


def test_second_turn_is_the_goal_verbatim() -> None:
    opt = _make(personas=["STAN"])
    _start(opt)
    _inject(opt)
    resp = _inject(opt)
    assert isinstance(resp, ControllableInjection)
    assert resp.value == GOAL_TEXT


def test_third_turn_is_declined_to_end_the_run() -> None:
    opt = _make(personas=["STAN"])
    _start(opt)
    _inject(opt)
    _inject(opt)
    assert isinstance(_inject(opt), ControllableNoInjection)


def test_one_persona_per_run_in_order() -> None:
    names = ["STAN", "DUDE", "AntiDAN"]
    expected = [p.render(None) for p in load_personas(names)]
    opt = _make(personas=names)
    for want in expected:
        _start(opt)
        assert _inject(opt).value == want
        assert _inject(opt).value == GOAL_TEXT
        _end(opt)


def test_done_only_after_the_last_persona() -> None:
    opt = _make(personas=["STAN", "DUDE"])
    _start(opt)
    _inject(opt)
    assert _end(opt).done is False
    _start(opt)
    _inject(opt)
    assert _end(opt).done is True


def test_defaults_to_all_thirteen_personas() -> None:
    opt = _make()
    done_after = 0
    while True:
        _start(opt)
        _inject(opt)
        done_after += 1
        if _end(opt).done:
            break
    assert done_after == 13


def test_generator_name_is_read_from_a_model_observable() -> None:
    obs = ObservableValue(
        observable=Observable(name="target_model", security_domain=USER_TAG),
        content="Claude",
    )
    opt = _make(observables=[obs], personas=["Dan_11_0"])
    _start(opt)
    value = _inject(opt).value
    assert "Claude" in value
    assert "{generator.name}" not in value


def test_explicit_generator_name_wins_over_observable() -> None:
    obs = ObservableValue(
        observable=Observable(name="target_model", security_domain=USER_TAG),
        content="Claude",
    )
    opt = _make(observables=[obs], personas=["Dan_11_0"], generator_name="Gemini")
    _start(opt)
    assert "Gemini" in _inject(opt).value


def test_never_injects_into_system_prompt() -> None:
    opt = _make(personas=["STAN"])
    _start(opt)
    assert isinstance(_inject(opt, "system_prompt"), ControllableNoInjection)


def test_declines_non_free_text_surfaces() -> None:
    opt = _make(personas=["STAN"])
    _start(opt)
    assert isinstance(
        _inject(opt, "tool_description", value_type="json"), ControllableNoInjection
    )


def test_declines_once_every_persona_is_exhausted() -> None:
    opt = _make(personas=["STAN"])
    _start(opt)
    _inject(opt)
    _end(opt)
    _start(opt)
    assert isinstance(_inject(opt), ControllableNoInjection)


def test_current_persona_tracks_progress_then_empties() -> None:
    opt = _make(personas=["STAN", "DUDE"])
    assert opt.current_persona.name == "STAN"
    _start(opt)
    _inject(opt)
    _end(opt)
    assert opt.current_persona.name == "DUDE"
    _start(opt)
    _inject(opt)
    _end(opt)
    assert opt.current_persona is None


def test_rejects_unknown_persona() -> None:
    with pytest.raises(ValueError, match="unknown persona"):
        DANPersonasOptimizer(personas=["Nope"])


def test_rejects_empty_persona_list() -> None:
    with pytest.raises(ValueError, match="at least one persona"):
        DANPersonasOptimizer(personas=[])


def test_initialize_resets_state_between_tasks() -> None:
    opt = _make(personas=["STAN", "DUDE"])
    _start(opt)
    _inject(opt)
    _end(opt)
    asyncio.run(
        opt.initialize(
            goal=Goal(description="something else"),
            controllables=[_ctrl()],
            observables=[],
            llm_client=MagicMock(),
        )
    )
    _start(opt)
    assert _inject(opt).value == load_personas(["STAN"])[0].render(None)
    assert _inject(opt).value == "something else"


def test_goal_follows_the_persona_on_the_same_surface() -> None:
    """A persona primes one conversation, so turn 2 must land on the surface
    that received the preamble -- not on whatever surface comes next."""
    opt = _make(personas=["STAN"])
    _start(opt)
    assert isinstance(_inject(opt, "user_query"), ControllableInjection)
    # a different surface in the same run must not receive the goal
    assert isinstance(_inject(opt, "db_lookup"), ControllableNoInjection)
    # the original surface still gets it
    resp = _inject(opt, "user_query")
    assert isinstance(resp, ControllableInjection)
    assert resp.value == GOAL_TEXT


def test_channel_pinning_resets_between_runs() -> None:
    opt = _make(personas=["STAN", "DUDE"])
    _start(opt)
    _inject(opt, "chan_a")
    _end(opt)
    _start(opt)
    # a fresh run may pin a different surface
    assert isinstance(_inject(opt, "chan_b"), ControllableInjection)


def test_single_message_mode_sends_persona_and_goal_together() -> None:
    """For targets that offer each surface once per run, a second turn never
    arrives, so the goal must ride along with the persona."""
    opt = _make(personas=["STAN"], mode="single_message")
    _start(opt)
    resp = _inject(opt)
    assert isinstance(resp, ControllableInjection)
    persona = load_personas(["STAN"])[0].render(None)
    assert resp.value == f"{persona}\n\n{GOAL_TEXT}"
    # nothing further this run
    assert isinstance(_inject(opt), ControllableNoInjection)


def test_single_message_mode_still_advances_personas() -> None:
    opt = _make(personas=["STAN", "DUDE"], mode="single_message")
    _start(opt)
    _inject(opt)
    assert _end(opt).done is False
    _start(opt)
    assert isinstance(_inject(opt), ControllableInjection)
    assert _end(opt).done is True


def test_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        DANPersonasOptimizer(mode="whatever")
