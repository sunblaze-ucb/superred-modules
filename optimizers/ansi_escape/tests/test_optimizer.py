"""Tests for AnsiEscapeOptimizer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from ansi_escape_optimizer import (
    ASKS,
    HIGH_LEVEL_TASKS,
    REPEAT_STUBS,
    UNESCAPE_STUBS,
    AnsiEscapeOptimizer,
    build_prompts,
)
from ansi_escape_optimizer._vendor.garak_ansi.ansi import (
    ESCAPED_PAYLOADS,
    LIVE_PAYLOADS,
)
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

USER = SecurityDomainTag("user")


def _ctrl(name="user_query", value_type="text"):
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _make(**kw):
    opt = AnsiEscapeOptimizer(**kw)
    import ansi_escape_optimizer.optimizer as mod

    mod.classify_controllables = AsyncMock(return_value={})
    asyncio.run(
        opt.initialize(
            goal=Goal(description="probe the terminal"),
            controllables=[_ctrl()],
            observables=[],
            llm_client=MagicMock(),
        )
    )
    return opt


def _start(o):
    asyncio.run(o.on_event(RunStartEvent(trajectory=MagicMock())))


def _pre(o, n="user_query"):
    return asyncio.run(
        o.on_event(ControllablePreCallEvent(controllable=_ctrl(n), request=""))
    )


def _end(o):
    return asyncio.run(o.on_event(RunEndEvent(evaluation=None)))


# --- upstream fidelity ------------------------------------------------------
def test_escaped_variant_reproduces_upstream_AnsiEscaped() -> None:
    """``AnsiEscaped.__init__``: asks x tasks, then escaped payloads x repeat stubs."""
    expected = [f"{a} {t}" for a in ASKS for t in HIGH_LEVEL_TASKS]
    expected += [f"{s} {p}" for p in ESCAPED_PAYLOADS for s in REPEAT_STUBS]
    assert build_prompts("escaped") == expected
    assert len(expected) == 51


def test_raw_variant_reproduces_upstream_AnsiRaw() -> None:
    """``AnsiRaw.__init__`` adds live payloads and the unescape stubs."""
    expected = [f"{a} {t}" for a in ASKS for t in HIGH_LEVEL_TASKS]
    expected += [f"{s} {p}" for p in LIVE_PAYLOADS for s in REPEAT_STUBS]
    expected += [f"{s} {p}" for p in ESCAPED_PAYLOADS for s in UNESCAPE_STUBS]
    assert build_prompts("raw") == expected
    assert len(expected) == 58


def test_raw_carries_live_escape_bytes_and_escaped_does_not() -> None:
    """The two variants differ in whether a real ESC byte reaches the target."""
    raw = "".join(build_prompts("raw"))
    esc = "".join(build_prompts("escaped"))
    assert "\x1b" in raw or "\x9b" in raw
    assert "\x1b" not in esc and "\x9b" not in esc


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="variant"):
        build_prompts("sideways")  # type: ignore[arg-type]


# --- sweep ------------------------------------------------------------------
def test_each_run_sends_the_next_prompt_and_the_sweep_terminates() -> None:
    opt = _make()
    seen = []
    for _ in range(len(opt.prompts)):
        _start(opt)
        resp = _pre(opt)
        assert isinstance(resp, ControllableInjection)
        seen.append(resp.value)
        done = _end(opt).done
    assert done is True
    assert seen == build_prompts("escaped")


def test_shuffle_is_seeded_and_keeps_the_same_set() -> None:
    a = AnsiEscapeOptimizer(shuffle=True, seed=3).prompts
    b = AnsiEscapeOptimizer(shuffle=True, seed=3).prompts
    c = AnsiEscapeOptimizer(shuffle=True, seed=4).prompts
    assert a == b and a != c
    assert sorted(a) == sorted(build_prompts("escaped"))


def test_only_one_surface_per_run_receives_a_prompt() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)
    assert isinstance(_pre(opt, "other_surface"), ControllableNoInjection)


def test_never_injects_the_system_prompt() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)


def test_post_call_returns_an_injection_decision_not_a_bare_event_response() -> None:
    """The channel type-checks PostCall responses; a bare EventResponse raises."""
    opt = _make()
    response = asyncio.run(
        opt.on_event(
            ControllablePostCallEvent(controllable=_ctrl(), request="q", answer="a")
        )
    )
    assert isinstance(response, ControllableNoInjection)


def test_the_goal_is_not_embedded_this_probe_is_goal_agnostic() -> None:
    """Upstream's prompts are fixed; nothing here reads the task Goal."""
    opt = _make()
    assert all("probe the terminal" not in p for p in opt.prompts)
