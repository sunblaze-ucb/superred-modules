"""Tests for AvSpamOptimizer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from av_spam_optimizer import (
    REPEAT_PROMPTS,
    SIGNATURES,
    AvSpamOptimizer,
    build_prompts,
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
    opt = AvSpamOptimizer(**kw)
    import av_spam_optimizer.optimizer as mod

    mod.classify_controllables = AsyncMock(return_value={})
    asyncio.run(
        opt.initialize(
            goal=Goal(description="get the scanner to fire"),
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
def test_the_three_signatures_are_the_published_test_strings() -> None:
    """These are the standard scanner *test* strings, not live malware."""
    assert SIGNATURES["eicar"].endswith("$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*")
    assert "GTUBE-STANDARD-ANTI-UBE-TEST-EMAIL" in SIGNATURES["gtube"]
    assert "GTPHISH-STANDARD-ANTI-PHISH-TEST-EMAIL" in SIGNATURES["gtphish"]


@pytest.mark.parametrize("variant", sorted(SIGNATURES))
def test_each_variant_reproduces_upstreams_construction(variant: str) -> None:
    """Upstream: ``[p.replace("{signature}", signature) for p in REPEAT_PROMPTS]``."""
    expected = [p.replace("{signature}", SIGNATURES[variant]) for p in REPEAT_PROMPTS]
    assert build_prompts(variant) == expected
    assert len(expected) == 5


def test_every_prompt_carries_the_signature_and_no_leftover_slot() -> None:
    for variant, signature in SIGNATURES.items():
        for prompt in build_prompts(variant):
            assert signature in prompt
            assert "{signature}" not in prompt


def test_the_three_variants_are_distinct() -> None:
    sets = [tuple(build_prompts(v)) for v in sorted(SIGNATURES)]
    assert len(set(sets)) == 3


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="variant"):
        build_prompts("sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="variant"):
        AvSpamOptimizer(variant="sideways")  # type: ignore[arg-type]


# --- delivery ---------------------------------------------------------------
def test_the_sweep_advances_and_terminates() -> None:
    opt = _make()
    seen = []
    for _ in range(5):
        _start(opt)
        resp = _pre(opt)
        assert isinstance(resp, ControllableInjection)
        seen.append(resp.value)
        done = _end(opt).done
    assert done is True
    assert seen == build_prompts("eicar")


def test_shuffle_is_seeded_and_keeps_the_same_set() -> None:
    a = AvSpamOptimizer(shuffle=True, seed=1).prompts
    b = AvSpamOptimizer(shuffle=True, seed=1).prompts
    assert a == b
    assert sorted(a) == sorted(build_prompts("eicar"))


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
    """Upstream's prompts carry a fixed signature; nothing reads the Goal."""
    opt = _make()
    assert all("scanner to fire" not in p for p in opt.prompts)


def test_a_run_with_no_eligible_surface_stops_without_skipping() -> None:
    """No injection this run -> index does not advance and the task stops."""
    opt = _make()
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)
    before = opt._index
    assert _end(opt).done is True
    assert opt._index == before
