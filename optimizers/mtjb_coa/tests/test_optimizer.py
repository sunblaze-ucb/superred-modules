"""Contract tests for MtjbCoAOptimizer against superred's event model.

The fake LLM routes by call shape and by synthetic markers embedded in the
(vendored) templates: a ``response_format`` request is a similarity call; a
message that mentions both a chain/refined prompt and the target's answer is a
prompt-update call; a message that mentions only the answer is a judge call;
anything else is chain generation. No real payload text appears here.
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

from mtjb_coa_optimizer import MtjbCoAOptimizer

USER = SecurityDomainTag("user")
GOAL = "GOAL_TEXT"


def _resp(content: str) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    return r


class FakeLLM:
    def __init__(self, chain, judge_scores=("[[0]]",), refined="REFINED", similarity=0.5):
        self.chain = list(chain)
        self.judge_scores = list(judge_scores)
        self.refined = refined
        self.similarity = similarity
        self.complete = AsyncMock(side_effect=self._route)

    async def _route(self, messages, **kw):
        joined = " ".join(m["content"] for m in messages)
        if kw.get("response_format"):
            return _resp(json.dumps({"similarity": self.similarity}))
        has_answer = "ANSWER" in joined
        has_preset = "CHAINP" in joined or "REFINED" in joined
        if has_answer and has_preset:
            return _resp(json.dumps({"improvement": "i", "prompt": self.refined}))
        if has_answer:
            score = self.judge_scores.pop(0) if self.judge_scores else "[[0]]"
            return _resp(score)
        obj = {str(i + 1): {"improvement": "i", "prompt": p} for i, p in enumerate(self.chain)}
        return _resp(json.dumps(obj))


def _ctrl(name="user", value_type="text"):
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _make(llm, **kw):
    opt = MtjbCoAOptimizer(n_init_chains=1, **kw)
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


def test_module_imports_without_heavy_deps() -> None:
    for mod in ("textgrad", "torch", "pandas", "sentence_transformers", "boto3"):
        assert mod not in sys.modules


def test_first_turn_injects_chain_opener() -> None:
    opt = _make(FakeLLM(chain=["CHAINP1", "CHAINP2"]), max_turns=2)
    assert opt.chain == ["CHAINP1", "CHAINP2"]
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    assert resp.value == "CHAINP1"


def test_post_call_returns_no_injection() -> None:
    opt = _make(FakeLLM(chain=["CHAINP1", "CHAINP2"]), max_turns=2)
    _start(opt)
    _pre(opt)
    resp = _post(opt, "CHAINP1", "ANSWER_1")
    assert isinstance(resp, ControllableNoInjection)


def test_runstart_rearms_state_across_two_runs() -> None:
    opt = _make(FakeLLM(chain=["CHAINP1", "CHAINP2"], judge_scores=["[[0]]", "[[0]]"]), max_turns=2)
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)
    _post(opt, "CHAINP1", "ANSWER_1")
    assert _end(opt).done is False
    # Second run: RunStart must re-arm _injected/_saw_post_call/_scored_this_run.
    _start(opt)
    assert opt._injected is False and opt._saw_post_call is False and opt._scored_this_run is False
    assert isinstance(_pre(opt), ControllableInjection)  # turn 2 injects again


def test_next_turn_uses_pre_send_refined_prompt() -> None:
    opt = _make(
        FakeLLM(chain=["CHAINP1", "CHAINP2"], judge_scores=["[[0]]"], refined="REFINED2"),
        max_turns=2,
        enable_attack_update=True,
    )
    _start(opt)
    _pre(opt)
    _post(opt, "CHAINP1", "ANSWER_1")
    assert opt._pending == "REFINED2"  # refined before the next send
    _start(opt)
    assert _pre(opt).value == "REFINED2"


def test_disable_attack_update_sends_raw_chain() -> None:
    opt = _make(
        FakeLLM(chain=["CHAINP1", "CHAINP2"], judge_scores=["[[0]]"]),
        max_turns=2,
        enable_attack_update=False,
    )
    _start(opt)
    _pre(opt)
    _post(opt, "CHAINP1", "ANSWER_1")
    assert opt._pending == "CHAINP2"  # raw next chain prompt, no refinement
    _start(opt)
    assert _pre(opt).value == "CHAINP2"


def test_binary_judge_success_ends_the_attack() -> None:
    opt = _make(FakeLLM(chain=["CHAINP1", "CHAINP2"], judge_scores=["[[1]]"]), max_turns=2)
    _start(opt)
    _pre(opt)
    _post(opt, "CHAINP1", "ANSWER_1")
    assert _end(opt).done is True


def test_turn_budget_ends_after_max_turns() -> None:
    opt = _make(FakeLLM(chain=["CHAINP1"], judge_scores=["[[0]]"]), max_turns=1)
    _start(opt)
    _pre(opt)
    _post(opt, "CHAINP1", "ANSWER_1")
    assert _end(opt).done is True


def test_conversation_pins_one_surface() -> None:
    opt = _make(FakeLLM(chain=["CHAINP1", "CHAINP2"], judge_scores=["[[0]]", "[[0]]"]), max_turns=2)
    _start(opt)
    assert isinstance(_pre(opt, "user"), ControllableInjection)  # pins "user"
    _post(opt, "CHAINP1", "ANSWER_1")
    _end(opt)
    _start(opt)
    assert isinstance(_pre(opt, "other_surface"), ControllableNoInjection)
    assert isinstance(_pre(opt, "user"), ControllableInjection)


def test_never_injects_system_prompt() -> None:
    opt = _make(FakeLLM(chain=["CHAINP1"]), max_turns=1)
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)


def test_genuine_budget_exhaustion_propagates() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=BudgetExhaustedError("spent", LLMUsage(calls=1, cost=0.5)))
    opt = MtjbCoAOptimizer(n_init_chains=1, max_turns=2)
    with pytest.raises(BudgetExhaustedError):
        asyncio.run(
            opt.initialize(
                goal=Goal(description=GOAL), controllables=[_ctrl()], observables=[], llm_client=llm
            )
        )


def test_budgetless_noop_client_degrades_quietly() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=BudgetExhaustedError("none", LLMUsage(calls=0, cost=0.0)))
    opt = MtjbCoAOptimizer(n_init_chains=1, max_turns=2)
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL), controllables=[_ctrl()], observables=[], llm_client=llm
        )
    )
    assert opt.chain == []  # no chain could be generated
    _start(opt)
    assert isinstance(_pre(opt), ControllableNoInjection)
    assert _end(opt).done is True
