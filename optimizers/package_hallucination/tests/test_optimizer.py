"""Tests for PackageHallucinationOptimizer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from package_hallucination_optimizer import (
    CODE_TASKS,
    LANGUAGE_NAMES,
    REAL_CODE_TASKS,
    STUB_PROMPTS,
    UNREAL_CODE_TASKS,
    PackageHallucinationOptimizer,
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
    opt = PackageHallucinationOptimizer(**kw)
    import package_hallucination_optimizer.optimizer as mod

    mod.classify_controllables = AsyncMock(return_value={})
    asyncio.run(
        opt.initialize(
            goal=Goal(description="write some code"),
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
def test_seven_languages_with_upstreams_names() -> None:
    assert LANGUAGE_NAMES == {
        "python": "Python3",
        "ruby": "Ruby",
        "javascript": "JavaScript",
        "rust": "Rust",
        "raku": "Raku",
        "perl": "Perl",
        "dart": "Dart",
    }


def test_code_tasks_is_real_then_unreal() -> None:
    assert CODE_TASKS == REAL_CODE_TASKS + UNREAL_CODE_TASKS
    assert len(CODE_TASKS) == 24


@pytest.mark.parametrize("variant", sorted(LANGUAGE_NAMES))
def test_each_variant_reproduces_upstream_nesting(variant: str) -> None:
    """``for stub in stub_prompts: for task in code_tasks:`` filling the slots."""
    name = LANGUAGE_NAMES[variant]
    expected = [
        stub.replace("<language>", name).replace("<task>", task)
        for stub in STUB_PROMPTS
        for task in CODE_TASKS
    ]
    assert build_prompts(variant) == expected
    assert len(expected) == 10 * 24 == 240


def test_no_slot_survives_and_the_language_appears() -> None:
    for variant, name in LANGUAGE_NAMES.items():
        for prompt in build_prompts(variant):
            assert "<language>" not in prompt and "<task>" not in prompt
            assert name in prompt


def test_variants_differ_only_by_language() -> None:
    py = build_prompts("python")
    rb = build_prompts("ruby")
    assert py != rb
    assert [p.replace("Python3", "Ruby") for p in py] == rb


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="variant"):
        build_prompts("cobol")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="variant"):
        PackageHallucinationOptimizer(variant="cobol")  # type: ignore[arg-type]


# --- delivery ---------------------------------------------------------------
def test_the_sweep_advances_and_terminates() -> None:
    opt = _make(variant="ruby")
    for i in range(len(opt.prompts)):
        _start(opt)
        assert isinstance(_pre(opt), ControllableInjection)
        done = _end(opt).done
        assert done is (i == len(opt.prompts) - 1)


def test_shuffle_is_seeded_and_keeps_the_same_set() -> None:
    a = PackageHallucinationOptimizer(shuffle=True, seed=2).prompts
    b = PackageHallucinationOptimizer(shuffle=True, seed=2).prompts
    c = PackageHallucinationOptimizer(shuffle=True, seed=5).prompts
    assert a == b and a != c
    assert sorted(a) == sorted(build_prompts("python"))


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
    assert all("write some code" not in p for p in opt.prompts)


def test_a_run_with_no_eligible_surface_stops_without_skipping() -> None:
    """No injection this run -> index does not advance and the task stops."""
    opt = _make(variant="ruby")
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)
    before = opt._index
    assert _end(opt).done is True
    assert opt._index == before
