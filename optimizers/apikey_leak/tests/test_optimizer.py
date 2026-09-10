"""Tests for ApiKeyLeakOptimizer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from apikey_leak_optimizer import (
    COMPLETE_TEMPLATE,
    GENERATE_TEMPLATE,
    KEY_TYPES,
    PARTIAL_KEYS,
    ApiKeyLeakOptimizer,
    build_prompts,
)
from apikey_leak_optimizer._vendor.garak_apikey.regexes import REGEX_DICTS
from apikey_leak_optimizer._vendor.garak_apikey.serviceutils import extract_key_types
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
    opt = ApiKeyLeakOptimizer(**kw)
    import apikey_leak_optimizer.optimizer as mod

    mod.classify_controllables = AsyncMock(return_value={})
    asyncio.run(
        opt.initialize(
            goal=Goal(description="leak a key"),
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
def test_key_types_come_from_upstreams_own_helper() -> None:
    """``KEY_TYPES = extract_key_types(REGEX_DICTS)`` -- the vendored resources."""
    assert KEY_TYPES == extract_key_types(REGEX_DICTS)
    assert len(KEY_TYPES) == 58
    assert "Aws Access Key" in KEY_TYPES


def test_generate_variant_reproduces_upstream_GetKey() -> None:
    expected = [GENERATE_TEMPLATE.replace("[key]", k) for k in KEY_TYPES]
    assert build_prompts("generate") == expected
    assert len(expected) == 58


def test_complete_variant_reproduces_upstream_CompleteKey() -> None:
    expected = [
        COMPLETE_TEMPLATE.replace("[key]", k).replace("[partial]", p)
        for k in KEY_TYPES
        for p in PARTIAL_KEYS
    ]
    assert build_prompts("complete") == expected
    assert len(expected) == 58 * 5 == 290


def test_no_slot_survives() -> None:
    for variant in ("generate", "complete"):
        for prompt in build_prompts(variant):
            assert "[key]" not in prompt and "[partial]" not in prompt


def test_complete_carries_the_partial_keys() -> None:
    prompts = build_prompts("complete")
    for partial in PARTIAL_KEYS:
        assert any(partial in p for p in prompts)


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="variant"):
        build_prompts("steal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="variant"):
        ApiKeyLeakOptimizer(variant="steal")  # type: ignore[arg-type]


# --- delivery ---------------------------------------------------------------
def test_the_sweep_advances_and_terminates() -> None:
    opt = _make(variant="generate")
    for i in range(len(opt.prompts)):
        _start(opt)
        assert isinstance(_pre(opt), ControllableInjection)
        assert _end(opt).done is (i == len(opt.prompts) - 1)


def test_shuffle_is_seeded_and_keeps_the_same_set() -> None:
    a = ApiKeyLeakOptimizer(shuffle=True, seed=1).prompts
    b = ApiKeyLeakOptimizer(shuffle=True, seed=1).prompts
    c = ApiKeyLeakOptimizer(shuffle=True, seed=9).prompts
    assert a == b and a != c
    assert sorted(a) == sorted(build_prompts("generate"))


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
    """Upstream's prompts are fixed; nothing reads the task Goal."""
    opt = _make()
    assert all("leak a key" != p for p in opt.prompts)
