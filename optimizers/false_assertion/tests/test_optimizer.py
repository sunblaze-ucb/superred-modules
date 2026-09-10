"""Tests for FalseAssertionOptimizer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from false_assertion_optimizer import (
    PREFIXES,
    FalseAssertionOptimizer,
    build_prompts,
    load_false_claims,
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
    opt = FalseAssertionOptimizer(**kw)
    import false_assertion_optimizer.optimizer as mod

    mod.classify_controllables = AsyncMock(return_value={})
    asyncio.run(
        opt.initialize(
            goal=Goal(description="debunk this"),
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
def test_fifty_claims_three_prefixes() -> None:
    assert PREFIXES == ["", "Why is it that", "Explain why"]
    assert len(load_false_claims()) == 50
    assert len(build_prompts()) == 150


def test_the_data_file_last_line_is_read_despite_no_trailing_newline() -> None:
    """The file has 49 newlines but 50 claims; the last must not be lost."""
    claims = load_false_claims()
    assert claims[-1].strip() == claims[-1]
    assert claims[-1]


def test_reproduces_upstream_construction() -> None:
    """``FalseAssertion.__init__``: bare claim, then each prefix with the
    leading-'The' lowercased."""
    claims = load_false_claims()
    expected = []
    for prefix in PREFIXES:
        for claim in claims:
            if not prefix:
                expected.append(claim)
            else:
                c = "t" + claim[1:] if claim.startswith("The") else claim
                expected.append(f"{prefix} {c}")
    assert build_prompts() == expected


def test_bare_prefix_leaves_the_claim_untouched() -> None:
    claims = load_false_claims()
    assert build_prompts()[: len(claims)] == claims


def test_leading_The_is_lowercased_only_under_a_prefix() -> None:
    """A claim starting 'The' becomes 'the' when a prefix is prepended, and
    stays 'The' in the bare framing."""
    the_claims = [c for c in load_false_claims() if c.startswith("The")]
    assert the_claims, "fixture should contain at least one 'The ...' claim"
    prompts = build_prompts()
    # bare framing keeps 'The'
    assert any(p.startswith("The") for p in prompts)
    # prefixed framing lowercases it
    assert any(p.startswith("Why is it that the") for p in prompts)
    assert not any(p.startswith("Why is it that The") for p in prompts)


# --- delivery ---------------------------------------------------------------
def test_the_sweep_advances_and_terminates() -> None:
    opt = _make()
    assert len(opt.prompts) == 150
    for i in range(len(opt.prompts)):
        _start(opt)
        assert isinstance(_pre(opt), ControllableInjection)
        assert _end(opt).done is (i == len(opt.prompts) - 1)


def test_shuffle_is_seeded_and_keeps_the_same_set() -> None:
    a = FalseAssertionOptimizer(shuffle=True, seed=1).prompts
    b = FalseAssertionOptimizer(shuffle=True, seed=1).prompts
    c = FalseAssertionOptimizer(shuffle=True, seed=8).prompts
    assert a == b and a != c
    assert sorted(a) == sorted(build_prompts())


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


def test_the_probe_is_goal_agnostic() -> None:
    """The claims are fixed; nothing reads the task Goal."""
    opt = _make()
    assert all("debunk this" != p for p in opt.prompts)
