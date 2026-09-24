"""Contract tests for MtjbMixOptimizer's event-loop adapter.

The engine bridge (which needs textgrad) is stubbed here so the adapter logic --
role validation, RunStart re-arm, PreCall injection, PostCall decision, RunEnd
done -- is testable offline. One test exercises the *real* degrade path: with
textgrad absent the engine cannot start and the optimizer ends without sending.
"""

from __future__ import annotations

import asyncio
import sys
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
from superred.core.types.llm import BudgetExhaustedError, LLMUsage
from superred.core.types.security_domain import SecurityDomainTag

from mtjb_mix_optimizer import VALID_ROLES, MtjbMixOptimizer

USER = SecurityDomainTag("user")
GOAL = "GOAL_TEXT"


def _ctrl(name="user", value_type="text"):
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _init(opt, llm=None):
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL),
            controllables=[_ctrl()],
            observables=[],
            llm_client=llm or MagicMock(),
        )
    )


def _start(o):
    asyncio.run(o.on_event(RunStartEvent(trajectory=MagicMock())))


def _pre(o, n="user"):
    return asyncio.run(o.on_event(ControllablePreCallEvent(controllable=_ctrl(n), request="")))


def _post(o, request, answer, n="user"):
    return asyncio.run(
        o.on_event(ControllablePostCallEvent(controllable=_ctrl(n), request=request, answer=answer))
    )


def _end(o):
    return asyncio.run(o.on_event(RunEndEvent(evaluation=None)))


def _stub(opt, start_prompt, observe_seq):
    """Replace the engine seams with async stubs (no textgrad needed)."""
    seq = list(observe_seq)

    async def fake_start(_behavior):
        return start_prompt

    async def fake_observe(_response):
        nxt = seq.pop(0) if seq else None
        if nxt is None:
            opt._done = True
        return nxt

    opt._start_engine = fake_start
    opt._observe = fake_observe


def test_invalid_role_raises() -> None:
    with pytest.raises(ValueError):
        MtjbMixOptimizer(generator="NotAFamily")
    with pytest.raises(ValueError):
        MtjbMixOptimizer(judge_and_flow="Interactive")  # not selectable


def test_valid_roles_accepted() -> None:
    for role in sorted(VALID_ROLES):
        MtjbMixOptimizer(generator=role, updater=role, judge_and_flow=role)


def test_module_imports_without_heavy_deps() -> None:
    for mod in ("textgrad", "torch", "pandas"):
        assert mod not in sys.modules


def test_first_turn_injects_engine_prompt() -> None:
    opt = MtjbMixOptimizer()
    _stub(opt, "P1", ["P2"])
    _init(opt)
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    assert resp.value == "P1"


def test_post_call_returns_no_injection_and_advances() -> None:
    opt = MtjbMixOptimizer()
    _stub(opt, "P1", ["P2"])
    _init(opt)
    _start(opt)
    _pre(opt)
    resp = _post(opt, "P1", "ANSWER")
    assert isinstance(resp, ControllableNoInjection)
    assert _end(opt).done is False
    _start(opt)
    assert _pre(opt).value == "P2"


def test_runstart_rearms_state() -> None:
    opt = MtjbMixOptimizer()
    _stub(opt, "P1", ["P2"])
    _init(opt)
    _start(opt)
    _pre(opt)
    _post(opt, "P1", "A")
    _end(opt)
    _start(opt)
    assert opt._injected is False and opt._saw_post_call is False and opt._scored_this_run is False


def test_engine_finishing_ends_the_attack() -> None:
    opt = MtjbMixOptimizer()
    _stub(opt, "P1", [None])  # engine returns no next prompt -> done
    _init(opt)
    _start(opt)
    _pre(opt)
    _post(opt, "P1", "A")
    assert _end(opt).done is True


def test_conversation_pins_one_surface() -> None:
    opt = MtjbMixOptimizer()
    _stub(opt, "P1", ["P2"])
    _init(opt)
    _start(opt)
    assert isinstance(_pre(opt, "user"), ControllableInjection)
    _post(opt, "P1", "A")
    _end(opt)
    _start(opt)
    assert isinstance(_pre(opt, "other"), ControllableNoInjection)
    assert isinstance(_pre(opt, "user"), ControllableInjection)


def test_never_injects_system_prompt() -> None:
    opt = MtjbMixOptimizer()
    _stub(opt, "P1", [])
    _init(opt)
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)


def test_degrades_cleanly_without_textgrad() -> None:
    """The REAL engine path: textgrad is absent, so the engine cannot start."""
    opt = MtjbMixOptimizer()
    _init(opt, llm=MagicMock())  # no stub: exercises the real _start_engine
    assert opt._done is True
    _start(opt)
    assert isinstance(_pre(opt), ControllableNoInjection)
    assert _end(opt).done is True


def test_budget_exhaustion_propagates_from_engine() -> None:
    opt = MtjbMixOptimizer()

    class _RaisingEngine:
        async def observe(self, _llm, _response):
            raise BudgetExhaustedError("spent", LLMUsage(calls=1, cost=0.5))

    opt._engine = _RaisingEngine()
    with pytest.raises(BudgetExhaustedError):
        asyncio.run(opt._observe("response"))


def test_budgetless_engine_degrades_quietly() -> None:
    opt = MtjbMixOptimizer()

    class _NoopEngine:
        async def observe(self, _llm, _response):
            raise BudgetExhaustedError("none", LLMUsage(calls=0, cost=0.0))

    opt._engine = _NoopEngine()
    assert asyncio.run(opt._observe("response")) is None
    assert opt._done is True
