"""Contract tests for MtjbXTeamingOptimizer.

The fake LLM routes by call shape and synthetic markers: a ``response_format``
request is the judge; a message that carries the selected strategy's persona is
a turn-prompt call; anything else is strategy generation. No real payload text
appears here. The TextGrad refine path is not exercised (env-gated separately);
offline, a would-be refine advances the plan.
"""

from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock

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

from mtjb_xteaming_optimizer import MtjbXTeamingOptimizer

USER = SecurityDomainTag("user")
GOAL = "GOAL_TEXT"
_STRATEGY = {
    "persona": "PERSONA_X",
    "context": "CTX",
    "approach": "APP",
    "conversation_plan": {"turn_1": "f1", "turn_2": "f2", "final_turn": "ff"},
}


def _resp(content: str) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    return r


class FakeLLM:
    def __init__(self, judge_scores):
        self.judge_scores = list(judge_scores)
        self.turn_counter = 0
        self.complete = AsyncMock(side_effect=self._route)

    async def _route(self, messages, **kw):
        joined = " ".join(m["content"] for m in messages)
        if kw.get("response_format"):
            s = self.judge_scores.pop(0) if self.judge_scores else 1
            return _resp(json.dumps({"score": s, "reason": "why"}))
        if "PERSONA_X" in joined:  # attacker system prompt carries the strategy
            self.turn_counter += 1
            return _resp(f"<conversation>TP{self.turn_counter}</conversation>")
        return _resp(json.dumps({f"Strategy_{i}": _STRATEGY for i in range(10)}))


def _ctrl(name="user", value_type="text"):
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _make(llm, **kw):
    opt = MtjbXTeamingOptimizer(**kw)
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL),
            controllables=[_ctrl()],
            observables=[],
            llm_client=llm,
        )
    )
    return opt


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


def test_module_imports_without_textgrad() -> None:
    for mod in ("textgrad", "torch", "pandas"):
        assert mod not in sys.modules


def test_strategy_selected_and_first_turn_injected() -> None:
    opt = _make(FakeLLM(judge_scores=[1]))
    assert opt.strategy is not None
    assert opt.strategy["persona"] == "PERSONA_X"
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    assert resp.value == "TP1"  # extracted from <conversation> tags


def test_post_call_returns_no_injection() -> None:
    opt = _make(FakeLLM(judge_scores=[1]))
    _start(opt)
    _pre(opt)
    resp = _post(opt, "TP1", "ANSWER")
    assert isinstance(resp, ControllableNoInjection)


def test_runstart_rearms_across_runs() -> None:
    opt = _make(FakeLLM(judge_scores=[1, 1]))
    _start(opt)
    _pre(opt)
    _post(opt, "TP1", "ANSWER")
    _end(opt)
    _start(opt)
    assert opt._injected is False and opt._saw_post_call is False and opt._scored_this_run is False
    assert isinstance(_pre(opt), ControllableInjection)


def test_judge_score_five_ends_the_attack() -> None:
    opt = _make(FakeLLM(judge_scores=[5]))
    _start(opt)
    _pre(opt)
    _post(opt, "TP1", "ANSWER")
    assert _end(opt).done is True


def test_plan_advances_turn_by_turn_offline() -> None:
    opt = _make(FakeLLM(judge_scores=[1, 1, 1]), max_total_turns=10)
    _start(opt)
    assert _pre(opt).value == "TP1"
    _post(opt, "TP1", "A1")
    assert _end(opt).done is False
    _start(opt)
    assert _pre(opt).value == "TP2"
    _post(opt, "TP2", "A2")
    _end(opt)
    _start(opt)
    assert _pre(opt).value == "TP3"  # final-turn prompt
    _post(opt, "TP3", "A3")
    assert _end(opt).done is True  # plan exhausted


def test_global_turn_cap_ends() -> None:
    opt = _make(FakeLLM(judge_scores=[1]), max_total_turns=1)
    _start(opt)
    _pre(opt)
    _post(opt, "TP1", "A1")
    assert _end(opt).done is True


def test_conversation_pins_one_surface() -> None:
    opt = _make(FakeLLM(judge_scores=[1, 1]))
    _start(opt)
    assert isinstance(_pre(opt, "user"), ControllableInjection)
    _post(opt, "TP1", "A1")
    _end(opt)
    _start(opt)
    assert isinstance(_pre(opt, "other"), ControllableNoInjection)
    assert isinstance(_pre(opt, "user"), ControllableInjection)


def test_never_injects_system_prompt() -> None:
    opt = _make(FakeLLM(judge_scores=[1]))
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)


def test_genuine_budget_exhaustion_propagates() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=BudgetExhaustedError("spent", LLMUsage(calls=1, cost=0.5)))
    opt = MtjbXTeamingOptimizer()
    with pytest.raises(BudgetExhaustedError):
        asyncio.run(
            opt.initialize(
                goal=Goal(description=GOAL), controllables=[_ctrl()], observables=[], llm_client=llm
            )
        )


def test_budgetless_noop_client_degrades_quietly() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=BudgetExhaustedError("none", LLMUsage(calls=0, cost=0.0)))
    opt = MtjbXTeamingOptimizer()
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL), controllables=[_ctrl()], observables=[], llm_client=llm
        )
    )
    assert opt.strategy is None
    _start(opt)
    assert isinstance(_pre(opt), ControllableNoInjection)
    assert _end(opt).done is True


def test_refine_falls_back_to_plan_without_textgrad() -> None:
    """With textgrad absent, a RETRY advances the plan rather than crashing."""
    # scores that force a RETRY at plan step 2 (score not beating best) then finish
    opt = _make(FakeLLM(judge_scores=[3, 1, 1]), max_total_turns=10, max_refines_per_turn=4)
    # step 1 -> continue
    _start(opt)
    _pre(opt)
    _post(opt, "TP1", "A1")
    _end(opt)
    _start(opt)
    assert _pre(opt).value == "TP2"
    _post(opt, "TP2", "A2")  # score 1 <= best 3 -> RETRY -> fallback advance
    # fell back to the plan: next injected prompt is a freshly planned turn
    _start(opt)
    assert _pre(opt).value == "TP3"
