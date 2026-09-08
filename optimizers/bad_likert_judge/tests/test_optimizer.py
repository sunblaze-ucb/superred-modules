"""Tests for BadLikertJudgeOptimizer's turn construction."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from bad_likert_judge_optimizer.optimizer import (
    DEFAULT_CATEGORY,
    DEFAULT_ENABLE_REFINEMENT,
    DEFAULT_MAX_BACKTRACKS,
    DEFAULT_NUM_TURNS,
    SUPPORTED_CATEGORIES,
    BadLikertJudgeOptimizer,
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

EXAMPLES = json.dumps({"score_1_example": "benign", "score_3_example": "SCORE3"})
REFINED = json.dumps({"refined_example": "REFINED3"})
ATTACK = json.dumps({"new_attack_prompt": "ATTACK_PROMPT"})
NO_REFUSAL = json.dumps({"refusal": False, "reasoning": "complied"})
REFUSAL = json.dumps({"refusal": True, "reasoning": "declined"})


def _ctrl(name: str = "user_query", value_type: str = "text") -> Controllable:
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _llm(*contents: str) -> MagicMock:
    llm = MagicMock()
    out = []
    for c in contents:
        r = MagicMock()
        r.choices = [MagicMock()]
        r.choices[0].message.content = c
        out.append(r)
    llm.complete = AsyncMock(side_effect=out)
    return llm


def _make(llm: MagicMock, **kw) -> BadLikertJudgeOptimizer:
    opt = BadLikertJudgeOptimizer(**kw)
    import bad_likert_judge_optimizer.optimizer as mod

    # Classification would consume a scripted llm.complete response; these
    # tests script the Likert chain exactly. Covered in test_surfaces.py.
    mod.classify_controllables = AsyncMock(return_value={})
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL),
            controllables=[_ctrl()],
            observables=[],
            llm_client=llm,
        )
    )
    return opt


def _start(opt): asyncio.run(opt.on_event(RunStartEvent(trajectory=MagicMock())))
def _pre(opt, name="user_query"):
    return asyncio.run(
        opt.on_event(ControllablePreCallEvent(controllable=_ctrl(name), request=""))
    )
def _end(opt): return asyncio.run(opt.on_event(RunEndEvent(evaluation=None)))


def test_upstream_defaults() -> None:
    """Pinned to DeepTeam's ``BadLikertJudge.__init__`` signature verbatim."""
    assert DEFAULT_CATEGORY == "bias"
    assert DEFAULT_NUM_TURNS == 5
    assert DEFAULT_ENABLE_REFINEMENT is True
    assert DEFAULT_MAX_BACKTRACKS == 3
    assert len(SUPPORTED_CATEGORIES) == 58


def test_constructor_defaults_match_the_upstream_constants() -> None:
    """A constructed optimizer uses upstream's values, not tuned-down ones."""
    opt = BadLikertJudgeOptimizer()
    assert opt._category == DEFAULT_CATEGORY
    assert opt._num_turns == DEFAULT_NUM_TURNS
    assert opt._enable_refinement is DEFAULT_ENABLE_REFINEMENT
    assert opt._max_backtracks == DEFAULT_MAX_BACKTRACKS


def test_a_turn_runs_the_three_simulator_steps_then_injects() -> None:
    """examples -> attack -> non_refusal, then the attack goes to the target."""
    llm = _llm(EXAMPLES, ATTACK, NO_REFUSAL)
    opt = _make(llm, enable_refinement=False)
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    assert resp.value == "ATTACK_PROMPT"
    assert llm.complete.await_count == 3  # refinement explicitly off here


def test_the_default_turn_runs_the_refine_step_too() -> None:
    """Upstream defaults enable_refinement on, so a default turn is four calls."""
    llm = _llm(EXAMPLES, REFINED, ATTACK, NO_REFUSAL)
    opt = _make(llm)
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)
    assert llm.complete.await_count == 4


def test_refinement_adds_the_refine_step_and_uses_its_output() -> None:
    llm = _llm(EXAMPLES, REFINED, ATTACK, NO_REFUSAL)
    opt = _make(llm, enable_refinement=True)
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)
    assert llm.complete.await_count == 4
    # the refined example must be what the attack step was given
    sent = llm.complete.await_args_list[2].args[0][0]["content"]
    assert "REFINED3" in sent and "SCORE3" not in sent


def test_refinement_failure_keeps_the_original_example() -> None:
    llm = _llm(EXAMPLES, "not json", ATTACK, NO_REFUSAL)
    opt = _make(llm, enable_refinement=True)
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)
    sent = llm.complete.await_args_list[2].args[0][0]["content"]
    assert "SCORE3" in sent


def test_simulator_refusal_spends_a_backtrack_and_sends_nothing() -> None:
    llm = _llm(EXAMPLES, ATTACK, REFUSAL)
    opt = _make(llm, enable_refinement=False)
    _start(opt)
    assert isinstance(_pre(opt), ControllableNoInjection)


def test_backtracks_exhaust_and_end_the_run() -> None:
    """max_backtracks consecutive refusals stop the attack."""
    llm = _llm(*([EXAMPLES, ATTACK, REFUSAL] * 2))
    opt = _make(llm, max_backtracks=2, enable_refinement=False)
    for _ in range(2):
        _start(opt)
        assert isinstance(_pre(opt), ControllableNoInjection)
    assert _end(opt).done is True


def test_unparseable_examples_step_is_a_backtrack() -> None:
    llm = _llm("garbage")
    opt = _make(llm, max_backtracks=1)
    _start(opt)
    assert isinstance(_pre(opt), ControllableNoInjection)
    assert _end(opt).done is True


def test_runs_for_num_turns_then_reports_done() -> None:
    llm = _llm(*([EXAMPLES, ATTACK, NO_REFUSAL] * 2))
    opt = _make(llm, num_turns=2, enable_refinement=False)
    for i in range(2):
        _start(opt)
        assert isinstance(_pre(opt), ControllableInjection)
        assert _end(opt).done is (i == 1)


def test_each_turn_feeds_the_previous_attack_back_in() -> None:
    """Upstream passes current_attack into likert_generate_examples, so the
    second turn escalates from the first rather than restarting from the goal."""
    llm = _llm(*([EXAMPLES, ATTACK, NO_REFUSAL] * 2))
    opt = _make(llm, num_turns=2, enable_refinement=False)
    _start(opt); _pre(opt); _end(opt)
    _start(opt)
    first = llm.complete.await_args_list[0].args[0][0]["content"]
    second = llm.complete.await_args_list[3].args[0][0]["content"]
    # current_attack advanced from the goal to the turn actually sent.
    assert "ATTACK_PROMPT" in second and "ATTACK_PROMPT" not in first
    # vulnerability_data still carries the goal every turn, as upstream does.
    assert f"Vulnerability: {GOAL}" in second


def test_first_turn_seeds_from_the_goal() -> None:
    llm = _llm(EXAMPLES, ATTACK, NO_REFUSAL)
    opt = _make(llm, enable_refinement=False)
    _start(opt)
    first = llm.complete.await_args_list[0].args[0][0]["content"]
    assert GOAL in first


def test_conversation_is_pinned_to_one_surface() -> None:
    llm = _llm(*([EXAMPLES, ATTACK, NO_REFUSAL] * 2))
    opt = _make(llm, num_turns=2, enable_refinement=False)
    _start(opt)
    assert isinstance(_pre(opt, "user_query"), ControllableInjection)
    _end(opt)
    _start(opt)
    assert isinstance(_pre(opt, "other_surface"), ControllableNoInjection)
    assert isinstance(_pre(opt, "user_query"), ControllableInjection)


def test_never_injects_system_prompt_or_non_free_text() -> None:
    llm = _llm(EXAMPLES, ATTACK, NO_REFUSAL)
    opt = _make(llm, enable_refinement=False)
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)


def test_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="unknown category"):
        BadLikertJudgeOptimizer(category="nope")
    with pytest.raises(ValueError, match="num_turns"):
        BadLikertJudgeOptimizer(num_turns=0)
    with pytest.raises(ValueError, match="max_backtracks"):
        BadLikertJudgeOptimizer(max_backtracks=0)


def test_genuine_budget_exhaustion_propagates_not_backtracks() -> None:
    """A spent budget must not be folded into the backtrack path: doing so
    burns every max_backtracks and reports done, which is indistinguishable
    from an attack the target defended -- a false zero."""
    from superred.core.types.llm import BudgetExhaustedError, LLMUsage

    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=BudgetExhaustedError("spent", LLMUsage(calls=7, cost=1.25))
    )
    opt = _make(llm)
    with pytest.raises(BudgetExhaustedError):
        _start(opt)


def test_budgetless_noop_client_still_degrades_to_a_backtrack() -> None:
    """The controller's budget-less noop client raises the same error with
    nothing spent; that one must stay a backtrack, not kill the run."""
    from superred.core.types.llm import BudgetExhaustedError, LLMUsage

    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=BudgetExhaustedError("no budget", LLMUsage(calls=0, cost=0.0))
    )
    opt = _make(llm, max_backtracks=1)
    _start(opt)
    assert isinstance(_pre(opt), ControllableNoInjection)
    assert _end(opt).done is True


def test_post_call_returns_an_injection_decision_not_a_bare_event_response() -> None:
    """The channel type-checks PostCall responses.

    A bare ``EventResponse`` raises ``TypeError`` and aborts the task on any
    target that emits post-call events. This attack never rewrites the
    target's answer, so it must decline.
    """
    opt = _make(_llm())
    response = asyncio.run(
        opt.on_event(
            ControllablePostCallEvent(
                controllable=_ctrl(), request="q", answer="a"
            )
        )
    )
    assert isinstance(response, ControllableNoInjection)
