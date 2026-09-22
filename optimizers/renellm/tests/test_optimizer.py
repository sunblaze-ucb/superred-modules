"""Contract tests for ReNeLLMOptimizer against superred's event model.

The fake LLM routes by call shape, using only *loaded* constants and synthetic
markers -- never any vendored template/prompt text:

* a message containing the vendored judge instruction (loaded as an identity
  constant, not reproduced here) is a *judge* call; among those, a message that
  also contains the synthetic reply marker is the reply score, otherwise it is
  the rewrite-still-harmful check;
* anything else is a rewrite operation, answered with a synthetic token.

No real payload text appears in this file.
"""

from __future__ import annotations

import asyncio
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

from renellm_optimizer import ReNeLLMOptimizer, vendored

USER = SecurityDomainTag("user")
GOAL = "SYNTHETIC_GOAL_ABC"
REWRITE_TOKEN = "REWRITTEN_XYZ"
REPLY_MARKER = "ZZ_REPLY_MARKER_ZZ"
# Loaded (never printed) so the fake LLM can recognise a judge call by identity.
HARM_JUDGE_PROMPT = vendored.load().harm_judge_prompt
SCENARIOS = vendored.load().scenarios


def _resp(content: str) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    return r


class FakeLLM:
    """Routes rewrite vs. rewrite-judge vs. reply-judge calls."""

    def __init__(
        self,
        *,
        rewrite: str = REWRITE_TOKEN,
        accept_rewrite: str = "1",
        reply_verdict: str = "1",
    ) -> None:
        self.rewrite = rewrite
        self.accept_rewrite = accept_rewrite
        self.reply_verdict = reply_verdict
        self.complete = AsyncMock(side_effect=self._route)

    async def _route(self, messages, **kwargs):
        joined = " ".join(m["content"] for m in messages)
        if HARM_JUDGE_PROMPT in joined:
            if REPLY_MARKER in joined:
                return _resp(self.reply_verdict)
            return _resp(self.accept_rewrite)
        return _resp(self.rewrite)


def _ctrl(name: str = "user", value_type: str = "text") -> Controllable:
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _make(llm, controllables=None, **kwargs) -> ReNeLLMOptimizer:
    opt = ReNeLLMOptimizer(seed=7, **kwargs)
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL),
            controllables=controllables if controllables is not None else [_ctrl()],
            observables=[],
            llm_client=llm,
        )
    )
    return opt


def _start(o) -> None:
    asyncio.run(o.on_event(RunStartEvent(trajectory=MagicMock())))


def _pre(o, name: str = "user", value_type: str = "text"):
    event = ControllablePreCallEvent(controllable=_ctrl(name, value_type), request="")
    return asyncio.run(o.on_event(event))


def _post(o, request: str, answer: str, name: str = "user"):
    event = ControllablePostCallEvent(controllable=_ctrl(name), request=request, answer=answer)
    return asyncio.run(o.on_event(event))


def _end(o, evaluation=None):
    return asyncio.run(o.on_event(RunEndEvent(evaluation=evaluation)))


def _harmful_reply() -> str:
    # Non-refusal synthetic reply carrying the routing marker.
    return f"{REPLY_MARKER} detailed synthetic content"


# ---------------------------------------------------------------------------
# imports / entry point
# ---------------------------------------------------------------------------
def test_module_imports_without_heavy_deps() -> None:
    for mod in ("torch", "transformers", "nltk", "sentence_transformers", "pandas"):
        assert mod not in sys.modules


def test_public_api_exposes_optimizer_subclass() -> None:
    # The module is discovered via its package public API (no entry-point group
    # exists in this repo): the class imports from the package root, is an
    # Optimizer subclass, and constructs with defaults.
    from superred.core.interfaces.optimizer import Optimizer

    from renellm_optimizer import ReNeLLMOptimizer as Cls

    assert issubclass(Cls, Optimizer)
    assert isinstance(Cls(), Optimizer)


# ---------------------------------------------------------------------------
# core contract
# ---------------------------------------------------------------------------
def test_pre_call_injects_and_post_call_declines() -> None:
    opt = _make(FakeLLM())
    _start(opt)
    pre = _pre(opt)
    assert isinstance(pre, ControllableInjection)
    post = _post(opt, pre.value, _harmful_reply())
    assert isinstance(post, ControllableNoInjection)


def test_rewrite_applied_and_scenario_nested_before_send() -> None:
    """A rewrite op transforms the goal AND a scenario wraps it, pre-send."""
    opt = _make(FakeLLM())
    _start(opt)
    injected = _pre(opt).value
    # The rewrite ran: original goal is gone, rewrite token present.
    assert GOAL not in injected
    assert REWRITE_TOKEN in injected
    # A scenario nested it: the payload is exactly one nested scenario, and is
    # strictly larger than the bare rewrite (the carrier wrapped it).
    nested = {s.replace("<>", REWRITE_TOKEN) for s in SCENARIOS}
    assert injected in nested
    assert injected != REWRITE_TOKEN


def test_judge_success_stops_the_attack() -> None:
    opt = _make(FakeLLM(reply_verdict="1"))
    _start(opt)
    pre = _pre(opt)
    _post(opt, pre.value, _harmful_reply())
    assert _end(opt).done is True


def test_judge_failure_continues_under_budget() -> None:
    opt = _make(FakeLLM(reply_verdict="0"), iter_max=5)
    _start(opt)
    pre = _pre(opt)
    _post(opt, pre.value, _harmful_reply())
    assert _end(opt).done is False


def test_refusal_short_circuits_the_judge() -> None:
    llm = FakeLLM(reply_verdict="1")
    opt = _make(llm, iter_max=5)
    _start(opt)
    pre = _pre(opt)
    calls_before = llm.complete.call_count
    # A refusal reply must not be judged (upstream short-circuit).
    _post(opt, pre.value, "I'm sorry, I cannot help with that.")
    end = _end(opt)
    assert end.done is False  # not scored a success
    assert llm.complete.call_count == calls_before  # no judge call at run end


def test_framework_success_is_authoritative() -> None:
    from superred.core.types.evaluation import EvaluationResult, Score

    opt = _make(FakeLLM(reply_verdict="0"))
    _start(opt)
    pre = _pre(opt)
    _post(opt, pre.value, _harmful_reply())
    evaluation = EvaluationResult(success=True, primary_score=Score(value=1.0))
    assert _end(opt, evaluation=evaluation).done is True


# ---------------------------------------------------------------------------
# RunStart re-arm
# ---------------------------------------------------------------------------
def test_runstart_rearms_state_across_two_runs() -> None:
    opt = _make(FakeLLM(reply_verdict="0"), iter_max=5)
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)
    _post(opt, "req", _harmful_reply())
    assert _end(opt).done is False
    # Second run: RunStart must reset per-run state and produce a fresh payload.
    _start(opt)
    assert opt._injected is False
    assert opt._pending is not None
    assert isinstance(_pre(opt), ControllableInjection)


def test_iter_max_budget_ends_the_attack() -> None:
    opt = _make(FakeLLM(reply_verdict="0"), iter_max=1)
    _start(opt)
    pre = _pre(opt)
    _post(opt, pre.value, _harmful_reply())
    assert _end(opt).done is True  # single-run budget exhausted


# ---------------------------------------------------------------------------
# surface selection
# ---------------------------------------------------------------------------
def test_never_injects_system_prompt() -> None:
    opt = _make(FakeLLM(), controllables=[_ctrl("system_prompt")])
    _start(opt)
    pre = _pre(opt, "system_prompt")
    assert isinstance(pre, ControllableNoInjection)


def test_no_eligible_surface_stops_without_spinning() -> None:
    opt = _make(FakeLLM(), controllables=[_ctrl("system_prompt")], iter_max=9)
    _start(opt)
    _pre(opt, "system_prompt")
    assert _end(opt).done is True


def test_non_free_text_surface_declined() -> None:
    opt = _make(FakeLLM(), controllables=[_ctrl("params", value_type="json")])
    _start(opt)
    assert isinstance(_pre(opt, "params", value_type="json"), ControllableNoInjection)


def test_target_controllable_name_pins_surface() -> None:
    opt = _make(
        FakeLLM(),
        controllables=[_ctrl("user"), _ctrl("other")],
        target_controllable_name="other",
    )
    _start(opt)
    assert isinstance(_pre(opt, "user"), ControllableNoInjection)
    assert isinstance(_pre(opt, "other"), ControllableInjection)


# ---------------------------------------------------------------------------
# budget handling
# ---------------------------------------------------------------------------
def test_genuine_budget_exhaustion_propagates() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=BudgetExhaustedError("spent", LLMUsage(calls=3, cost=0.5)))
    opt = ReNeLLMOptimizer(seed=1)
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL), controllables=[_ctrl()], observables=[], llm_client=llm
        )
    )
    with pytest.raises(BudgetExhaustedError):
        _start(opt)


def test_budgetless_noop_client_degrades_quietly() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=BudgetExhaustedError("none", LLMUsage(calls=0, cost=0.0)))
    opt = ReNeLLMOptimizer(seed=1)
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL), controllables=[_ctrl()], observables=[], llm_client=llm
        )
    )
    _start(opt)
    assert opt._pending is None  # no payload could be produced
    assert isinstance(_pre(opt), ControllableNoInjection)
    assert _end(opt).done is True
