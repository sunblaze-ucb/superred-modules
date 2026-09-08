"""Tests for JailbrokenOptimizer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from jailbroken_optimizer import TEMPLATE_COUNT, JailbrokenOptimizer, render
from jailbroken_optimizer._vendor.aig_jailbroken.template import JAILBROKEN_TEMPLATES
from superred.core.types.controllable import Controllable
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
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
    opt = JailbrokenOptimizer(**kw)
    import jailbroken_optimizer.optimizer as mod

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


def _start(o): asyncio.run(o.on_event(RunStartEvent(trajectory=MagicMock())))
def _pre(o, n="user_query"):
    return asyncio.run(o.on_event(ControllablePreCallEvent(controllable=_ctrl(n), request="")))
def _end(o): return asyncio.run(o.on_event(RunEndEvent(evaluation=None)))


def test_template_count_matches_upstream() -> None:
    assert TEMPLATE_COUNT == len(JAILBROKEN_TEMPLATES) == 18


def test_render_fills_the_query_slot() -> None:
    out = render(GOAL, 0)
    assert GOAL in out
    assert "{query}" not in out


def test_render_matches_upstream_enhance_shape() -> None:
    """Upstream: next(iter(template.values())).format(query=attack)."""
    for i in range(TEMPLATE_COUNT):
        expected = next(iter(JAILBROKEN_TEMPLATES[i].values())).format(query=GOAL)
        assert render(GOAL, i) == expected


def test_render_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="index"):
        render(GOAL, TEMPLATE_COUNT)


def test_first_run_injects_a_filled_template() -> None:
    opt = _make()
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    assert GOAL in resp.value


def test_sweeps_all_templates_one_per_run() -> None:
    opt = _make()
    seen = []
    while True:
        _start(opt)
        r = _pre(opt)
        if isinstance(r, ControllableNoInjection):
            break
        seen.append(r.value)
        if _end(opt).done:
            break
    assert len(seen) == TEMPLATE_COUNT
    # Upstream ships 18 templates but two are identical, so 17 distinct
    # renders. Preserved as-is (byte-identical vendoring).
    assert len(set(seen)) == 17


def test_order_is_seed_reproducible() -> None:
    a, b = _make(seed=7), _make(seed=7)
    _start(a); _start(b)
    assert _pre(a).value == _pre(b).value


def test_different_seeds_differ() -> None:
    a, b = _make(seed=0), _make(seed=1)
    _start(a); _start(b)
    # extremely unlikely to pick the same first template across 18 with diff seeds
    assert _pre(a).value != _pre(b).value


def test_no_shuffle_uses_file_order() -> None:
    opt = _make(shuffle=False)
    _start(opt)
    assert _pre(opt).value == render(GOAL, 0)


def test_injects_once_per_run() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_pre(opt, "user_query"), ControllableInjection)
    assert isinstance(_pre(opt, "db_lookup"), ControllableNoInjection)


def test_never_injects_system_prompt() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)


def test_declines_after_all_templates_spent() -> None:
    opt = _make()
    for _ in range(TEMPLATE_COUNT):
        _start(opt); _pre(opt); _end(opt)
    _start(opt)
    assert isinstance(_pre(opt), ControllableNoInjection)
