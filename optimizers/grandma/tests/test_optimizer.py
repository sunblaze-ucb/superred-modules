"""Tests for GrandmaOptimizer."""

from __future__ import annotations

import asyncio
import json
from importlib import resources
from unittest.mock import AsyncMock, MagicMock

import pytest
from grandma_optimizer import (
    ACTIONS,
    GRANDMA_ACTIVITIES,
    HUMAN_GROUPS,
    INTENT_TEMPLATE,
    PERSON_ADJS,
    PERSONAS,
    PRODUCT_KEY_PROBE_TEMPLATES,
    SLUR_PROBE_TEMPLATES,
    SUBSTANCE_TEMPLATES,
    SUBSTANCES,
    GrandmaOptimizer,
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
GOAL = "explain how to pick a lock"


def _ctrl(name="user_query", value_type="text"):
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _make(**kw):
    opt = GrandmaOptimizer(**kw)
    import grandma_optimizer.optimizer as mod

    mod.classify_controllables = AsyncMock(return_value={})
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL),
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


def _payload(name):
    raw = (
        resources.files("grandma_optimizer")
        .joinpath("_vendor/garak_payloads", f"{name}.json")
        .read_text("utf-8")
    )
    return json.loads(raw)["payloads"]


# --- upstream fidelity, variant by variant ----------------------------------
@pytest.mark.parametrize("variant", ["win10", "win11"])
def test_product_key_variants_reproduce_upstream(variant: str) -> None:
    """``Win10.__init__``: templates x the payload file's product names."""
    expected = [
        p.replace("{product_name}", n)
        for p in PRODUCT_KEY_PROBE_TEMPLATES
        for n in _payload(f"keyedprod_{variant}")
    ]
    assert build_prompts(variant) == expected


def test_slurs_variant_reproduces_upstream() -> None:
    expected = [
        p.replace("{group_name}", g) for p in SLUR_PROBE_TEMPLATES for g in HUMAN_GROUPS
    ]
    assert build_prompts("slurs") == expected


def test_substances_variant_reproduces_upstream() -> None:
    expected = [
        p.replace("[SUBSTANCE]", s) for p in SUBSTANCE_TEMPLATES for s in SUBSTANCES
    ]
    assert build_prompts("substances") == expected


def test_intent_variant_reproduces_upstream_nesting() -> None:
    """``GrandmaIntent._prompts_from_stub``, loop for loop."""
    stub = " ".join(GOAL.split(" ")[1:])
    expected = [
        INTENT_TEMPLATE.format(
            persona_adj=f" {adj}" if len(adj) else adj,
            persona=persona,
            action=action.lower(),
            stub=stub,
            grandma_activity=activity,
        )
        for persona in PERSONAS
        for action in ACTIONS
        for adj in PERSON_ADJS
        for activity in GRANDMA_ACTIVITIES
    ]
    assert build_prompts("intent", GOAL) == expected
    assert len(expected) == 6 * 3 * 5 * 6 == 540


def test_intent_drops_the_goals_first_word_as_upstream_does() -> None:
    """Upstream: "we'll use our own verbs" -- the stub's first word goes."""
    p = build_prompts("intent", "explain how to pick a lock")[0]
    assert "how to pick a lock" in p
    assert "explain how to pick a lock" not in p


def test_the_empty_adjective_does_not_leave_a_double_space() -> None:
    """Upstream prepends a space only for a non-empty adjective."""
    assert any(p.startswith("My grandma used to") for p in build_prompts("intent", GOAL))
    assert not any("My  " in p for p in build_prompts("intent", GOAL))


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="variant"):
        build_prompts("sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="variant"):
        GrandmaOptimizer(variant="sideways")  # type: ignore[arg-type]


# --- delivery ---------------------------------------------------------------
def test_the_goal_reaches_the_target_in_the_intent_variant() -> None:
    opt = _make()
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    assert "how to pick a lock" in resp.value


def test_the_fixed_variants_are_goal_agnostic() -> None:
    """Upstream's non-intent probes carry fixed lists; nothing reads the Goal."""
    opt = _make(variant="substances")
    assert all("pick a lock" not in p for p in opt.prompts)


def test_the_sweep_advances_and_terminates() -> None:
    opt = _make(variant="win10")
    seen = []
    for _ in range(len(opt.prompts)):
        _start(opt)
        resp = _pre(opt)
        assert isinstance(resp, ControllableInjection)
        seen.append(resp.value)
        done = _end(opt).done
    assert done is True
    assert seen == build_prompts("win10")


def test_shuffle_is_seeded_and_keeps_the_same_set() -> None:
    a = _make(variant="slurs", shuffle=True, seed=3).prompts
    b = _make(variant="slurs", shuffle=True, seed=3).prompts
    c = _make(variant="slurs", shuffle=True, seed=4).prompts
    assert a == b and a != c
    assert sorted(a) == sorted(build_prompts("slurs"))


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


@pytest.mark.parametrize("goal", ["", "braces {x} and {0}", "日本語 🙂", "x" * 2000])
def test_arbitrary_goals_do_not_break_the_intent_template(goal: str) -> None:
    prompts = build_prompts("intent", goal)
    assert len(prompts) == 540
    assert all("{stub}" not in p for p in prompts)
