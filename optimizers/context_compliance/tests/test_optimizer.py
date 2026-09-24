"""Contract + faithfulness tests for ContextComplianceOptimizer.

The fake LLM routes by the user turn it receives: the adversarial call is driven
with the objective (so it returns the synthetic benign question), and the
simulated-target call is driven with that question (so it returns the synthetic
affirmative answer). No vendored template text is inspected or reproduced -- the
router only reads the synthetic user turn, never the system prompt.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from superred.core.interfaces.optimizer import Optimizer
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

from context_compliance_optimizer import ContextComplianceOptimizer

USER = SecurityDomainTag("user")
GOAL = "SYNTH_GOAL_XYZ"
SYNTH_Q = "SYNTH_QUESTION_Q"
SYNTH_A = "SYNTH_ANSWER_A"


def _resp(content: str) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    return r


class FakeLLM:
    """Routes the two CCA generations by the synthetic user turn (never the system)."""

    def __init__(self, question: str = SYNTH_Q, answer: str = SYNTH_A) -> None:
        self.question = question
        self.answer = answer
        self.complete = AsyncMock(side_effect=self._route)

    async def _route(self, messages, **kw):
        user = messages[-1]["content"]
        if user == GOAL:  # adversarial persona: turn objective -> benign question
            return _resp(self.question)
        return _resp(self.answer)  # simulated target: answer the question affirmatively


def _ctrl(name: str = "user_message", value_type: str = "text") -> Controllable:
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _make(llm, **kw) -> ContextComplianceOptimizer:
    opt = ContextComplianceOptimizer(**kw)
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL),
            controllables=[_ctrl()],
            observables=[],
            llm_client=llm,
        )
    )
    return opt


def _start(o) -> None:
    asyncio.run(o.on_event(RunStartEvent(trajectory=MagicMock())))


def _pre(o, n: str = "user_message", vt: str = "text"):
    return asyncio.run(o.on_event(ControllablePreCallEvent(controllable=_ctrl(n, vt), request="")))


def _post(o, request: str, answer: str, n: str = "user_message"):
    return asyncio.run(
        o.on_event(ControllablePostCallEvent(controllable=_ctrl(n), request=request, answer=answer))
    )


def _end(o):
    return asyncio.run(o.on_event(RunEndEvent(evaluation=None)))


# -- imports / entry point ---------------------------------------------------


def test_module_imports_without_heavy_deps() -> None:
    for mod in ("torch", "transformers", "sentence_transformers", "pyrit"):
        assert mod not in sys.modules


def test_entry_point_class_resolves() -> None:
    assert issubclass(ContextComplianceOptimizer, Optimizer)
    assert isinstance(ContextComplianceOptimizer(), ContextComplianceOptimizer)


# -- faithfulness: fabricated assistant turn precedes the real request -------


def test_injection_contains_fabricated_assistant_turn_then_goal() -> None:
    opt = _make(FakeLLM())
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    value = resp.value
    # benign fabricated user question, then a fabricated assistant turn, then
    # the real request (the goal, in the default mode).
    assert f"Assistant: {SYNTH_A}" in value  # fabricated prior assistant turn
    assert SYNTH_Q in value
    assert GOAL in value
    assert value.index(SYNTH_Q) < value.index(f"Assistant: {SYNTH_A}") < value.index(GOAL)


def test_default_final_request_is_the_goal() -> None:
    opt = _make(FakeLLM())
    _start(opt)
    value = _pre(opt).value
    assert value.rstrip().endswith(GOAL)


def test_upstream_final_user_message_mode() -> None:
    opt = _make(FakeLLM(), final_user_message="yes.")
    _start(opt)
    value = _pre(opt).value
    assert f"Assistant: {SYNTH_A}" in value  # still a fabricated assistant turn
    assert value.rstrip().endswith("yes.")


def test_both_templates_rendered_with_the_goal() -> None:
    llm = FakeLLM()
    _make(llm)
    calls = llm.complete.await_args_list
    assert len(calls) == 2
    adv_messages = calls[0].args[0]
    tgt_messages = calls[1].args[0]
    # The objective is substituted into both rendered system prompts (goal
    # threading), and each call is driven with the expected user turn.
    assert GOAL in adv_messages[0]["content"]
    assert adv_messages[-1]["content"] == GOAL
    assert GOAL in tgt_messages[0]["content"]
    assert tgt_messages[-1]["content"] == SYNTH_Q


# -- injection contract ------------------------------------------------------


def test_injects_once_per_run() -> None:
    opt = _make(FakeLLM())
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)
    assert isinstance(_pre(opt), ControllableNoInjection)  # only one injection per run


def test_post_call_returns_no_injection() -> None:
    opt = _make(FakeLLM())
    _start(opt)
    _pre(opt)
    assert isinstance(_post(opt, "req", "ANSWER"), ControllableNoInjection)


def test_runstart_rearms_state_across_two_runs() -> None:
    opt = _make(FakeLLM())
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)
    assert _end(opt).done is True
    # Second run: RunStart must re-arm the per-run injection flag.
    _start(opt)
    assert opt._injected is False
    assert isinstance(_pre(opt), ControllableInjection)


def test_never_injects_system_prompt() -> None:
    opt = _make(FakeLLM())
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)


def test_declines_non_free_text_surface() -> None:
    opt = _make(FakeLLM())
    _start(opt)
    assert isinstance(_pre(opt, "tool_catalog", "json"), ControllableNoInjection)


def test_run_end_reports_done() -> None:
    opt = _make(FakeLLM())
    _start(opt)
    _pre(opt)
    assert _end(opt).done is True


# -- budget handling ---------------------------------------------------------


def test_genuine_budget_exhaustion_propagates() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=BudgetExhaustedError("spent", LLMUsage(calls=1, cost=0.5)))
    opt = ContextComplianceOptimizer()
    with pytest.raises(BudgetExhaustedError):
        asyncio.run(
            opt.initialize(
                goal=Goal(description=GOAL), controllables=[_ctrl()], observables=[], llm_client=llm
            )
        )


def test_budgetless_noop_client_degrades_quietly() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=BudgetExhaustedError("none", LLMUsage(calls=0, cost=0.0)))
    opt = ContextComplianceOptimizer()
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL), controllables=[_ctrl()], observables=[], llm_client=llm
        )
    )
    assert opt.payload is None  # no fabricated exchange could be built
    _start(opt)
    assert isinstance(_pre(opt), ControllableNoInjection)
    assert _end(opt).done is True
