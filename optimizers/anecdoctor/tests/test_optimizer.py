"""Contract tests for AnecdoctorOptimizer against superred's event model.

Every test uses synthetic neutral placeholders and a mocked LLM. The vendored
framing templates are replaced with synthetic stand-ins so no template body is
ever loaded into a fixture or an assertion (one separate test confirms the real
templates load and expose the ``{language}``/``{type}`` placeholders, without
printing their contents).
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

from anecdoctor_optimizer import AnecdoctorOptimizer, vendored

USER = SecurityDomainTag("user")
GOAL = "GOAL_CLAIM_TEXT"

# Synthetic framing stand-ins keyed by vendored filename; each keeps the upstream
# format placeholders so ``.format()`` behaves like the real templates.
_FAKE_FRAMING = {
    vendored.BUILD_KG_YAML: "KG_BUILD_FRAMING {language}",
    vendored.USE_KG_YAML: "USE_KG_FRAMING {language} {type}",
    vendored.USE_FEWSHOT_YAML: "FEWSHOT_FRAMING {language} {type}",
}


def _resp(content: str) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    return r


class FakeLLM:
    """Records calls and returns a fixed synthetic knowledge-graph string."""

    def __init__(self, kg_response: str = "SYNTH_KG_GRAPH") -> None:
        self.kg_response = kg_response
        self.messages: list[list[dict[str, str]]] = []
        self.complete = AsyncMock(side_effect=self._route)

    async def _route(self, messages, **kw):
        self.messages.append(messages)
        return _resp(self.kg_response)


def _patch_framing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vendored, "load_prompt", lambda name: _FAKE_FRAMING[name])


def _ctrl(name: str = "user", value_type: str = "text") -> Controllable:
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _make(monkeypatch: pytest.MonkeyPatch, llm, **kw) -> AnecdoctorOptimizer:
    _patch_framing(monkeypatch)
    opt = AnecdoctorOptimizer(**kw)
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL),
            controllables=[_ctrl()],
            observables=[],
            llm_client=llm,
        )
    )
    return opt


def _start(o: AnecdoctorOptimizer) -> None:
    asyncio.run(o.on_event(RunStartEvent(trajectory=MagicMock())))


def _pre(o: AnecdoctorOptimizer, n: str = "user"):
    return asyncio.run(o.on_event(ControllablePreCallEvent(controllable=_ctrl(n), request="")))


def _post(o: AnecdoctorOptimizer, n: str = "user"):
    return asyncio.run(
        o.on_event(ControllablePostCallEvent(controllable=_ctrl(n), request="", answer="ANSWER"))
    )


def _end(o: AnecdoctorOptimizer):
    return asyncio.run(o.on_event(RunEndEvent(evaluation=None)))


# --------------------------------------------------------------------------- #
# imports / entry-point
# --------------------------------------------------------------------------- #
def test_module_imports_without_heavy_deps() -> None:
    for mod in ("torch", "transformers", "sentence_transformers", "nltk", "networkx"):
        assert mod not in sys.modules


def test_entry_point_class_resolves() -> None:
    assert issubclass(AnecdoctorOptimizer, Optimizer)


# --------------------------------------------------------------------------- #
# few-shot mode
# --------------------------------------------------------------------------- #
def test_fewshot_mode_injects_framed_request_without_llm(monkeypatch) -> None:
    llm = FakeLLM()
    opt = _make(monkeypatch, llm, use_knowledge_graph=False, example_claims=["SYNTH_CLAIM"])
    assert llm.complete.await_count == 0  # few-shot needs no attacker LLM
    assert opt.knowledge_graph_built is False
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    # framed request folds framing + the few-shot example block, goal-first.
    assert "FEWSHOT_FRAMING" in resp.value
    assert "### examples" in resp.value
    assert GOAL in resp.value  # goal folded in as the lead claim
    assert "SYNTH_CLAIM" in resp.value


def test_default_synthetic_claims_used_when_none_supplied(monkeypatch) -> None:
    from anecdoctor_optimizer import claims as claims_mod

    opt = _make(monkeypatch, FakeLLM(), use_knowledge_graph=False)
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    assert claims_mod.DEFAULT_EXAMPLE_CLAIMS[0] in resp.value


def test_include_goal_as_claim_false_omits_goal(monkeypatch) -> None:
    opt = _make(
        monkeypatch,
        FakeLLM(),
        use_knowledge_graph=False,
        example_claims=["SYNTH_CLAIM"],
        include_goal_as_claim=False,
    )
    assert GOAL not in opt.payload
    assert "SYNTH_CLAIM" in opt.payload


# --------------------------------------------------------------------------- #
# knowledge-graph mode
# --------------------------------------------------------------------------- #
def test_kg_mode_builds_kg_then_emits_framed_request(monkeypatch) -> None:
    llm = FakeLLM(kg_response="SYNTH_KG_GRAPH")
    opt = _make(monkeypatch, llm, use_knowledge_graph=True, example_claims=["SYNTH_CLAIM"])

    # The KG was built via exactly the attacker LLM.
    assert llm.complete.await_count >= 1
    assert opt.knowledge_graph_built is True

    # The KG-build call fed the claims to the model under the KG-build framing.
    build_msgs = llm.messages[0]
    roles = {m["role"]: m["content"] for m in build_msgs}
    assert "KG_BUILD_FRAMING" in roles["system"]
    assert "### examples" in roles["user"]
    assert GOAL in roles["user"] and "SYNTH_CLAIM" in roles["user"]

    # The framed request wraps the *knowledge graph*, not the raw few-shot block.
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    assert "USE_KG_FRAMING" in resp.value
    assert "SYNTH_KG_GRAPH" in resp.value


def test_kg_mode_falls_back_to_fewshot_when_kg_empty(monkeypatch) -> None:
    llm = FakeLLM(kg_response="   ")  # empty/blank KG response
    opt = _make(monkeypatch, llm, use_knowledge_graph=True, example_claims=["SYNTH_CLAIM"])
    assert opt.knowledge_graph_built is False
    assert "FEWSHOT_FRAMING" in opt.payload
    assert "SYNTH_CLAIM" in opt.payload


# --------------------------------------------------------------------------- #
# event contract
# --------------------------------------------------------------------------- #
def test_post_call_returns_no_injection(monkeypatch) -> None:
    opt = _make(monkeypatch, FakeLLM(), use_knowledge_graph=False)
    _start(opt)
    _pre(opt)
    assert isinstance(_post(opt), ControllableNoInjection)


def test_run_end_is_done(monkeypatch) -> None:
    opt = _make(monkeypatch, FakeLLM(), use_knowledge_graph=False)
    _start(opt)
    _pre(opt)
    assert _end(opt).done is True


def test_only_injects_once_per_run(monkeypatch) -> None:
    opt = _make(monkeypatch, FakeLLM(), use_knowledge_graph=False)
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)
    assert isinstance(_pre(opt), ControllableNoInjection)  # already injected this run


def test_never_injects_system_prompt(monkeypatch) -> None:
    opt = _make(monkeypatch, FakeLLM(), use_knowledge_graph=False)
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)


def test_declines_non_free_text_surface(monkeypatch) -> None:
    _patch_framing(monkeypatch)
    opt = AnecdoctorOptimizer(use_knowledge_graph=False)
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL),
            controllables=[_ctrl("payload", value_type="json")],
            observables=[],
            llm_client=FakeLLM(),
        )
    )
    _start(opt)
    resp = asyncio.run(
        opt.on_event(ControllablePreCallEvent(controllable=_ctrl("payload", "json"), request=""))
    )
    assert isinstance(resp, ControllableNoInjection)


def test_runstart_rearms_state_across_two_runs(monkeypatch) -> None:
    opt = _make(monkeypatch, FakeLLM(), use_knowledge_graph=False, example_claims=["SYNTH_CLAIM"])
    _start(opt)
    first = _pre(opt)
    assert isinstance(first, ControllableInjection)
    _post(opt)
    assert _end(opt).done is True
    # Second run: RunStart must re-arm _injected so the request is emitted again.
    _start(opt)
    assert opt._injected is False
    second = _pre(opt)
    assert isinstance(second, ControllableInjection)
    assert second.value == first.value  # same deterministic framed request


# --------------------------------------------------------------------------- #
# budget behaviour (KG mode is the only path that calls the LLM)
# --------------------------------------------------------------------------- #
def test_genuine_budget_exhaustion_propagates(monkeypatch) -> None:
    _patch_framing(monkeypatch)
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=BudgetExhaustedError("spent", LLMUsage(calls=1, cost=0.5)))
    opt = AnecdoctorOptimizer(use_knowledge_graph=True)
    with pytest.raises(BudgetExhaustedError):
        asyncio.run(
            opt.initialize(
                goal=Goal(description=GOAL),
                controllables=[_ctrl()],
                observables=[],
                llm_client=llm,
            )
        )


def test_budgetless_noop_client_degrades_to_fewshot(monkeypatch) -> None:
    _patch_framing(monkeypatch)
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=BudgetExhaustedError("none", LLMUsage(calls=0, cost=0.0)))
    opt = AnecdoctorOptimizer(use_knowledge_graph=True, example_claims=["SYNTH_CLAIM"])
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL),
            controllables=[_ctrl()],
            observables=[],
            llm_client=llm,
        )
    )
    assert opt.knowledge_graph_built is False  # KG could not be built
    assert "FEWSHOT_FRAMING" in opt.payload and "SYNTH_CLAIM" in opt.payload
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)  # still emits a request


# --------------------------------------------------------------------------- #
# real vendored templates (structural only; no bodies printed)
# --------------------------------------------------------------------------- #
def test_real_templates_load_with_placeholders() -> None:
    build_kg = vendored.load_prompt(vendored.BUILD_KG_YAML)
    use_fewshot = vendored.load_prompt(vendored.USE_FEWSHOT_YAML)
    use_kg = vendored.load_prompt(vendored.USE_KG_YAML)
    assert "{language}" in build_kg
    assert "{language}" in use_fewshot and "{type}" in use_fewshot
    assert "{language}" in use_kg and "{type}" in use_kg
    # format() must succeed with the upstream placeholders (guards initialize()).
    assert build_kg.format(language="L")
    assert use_fewshot.format(language="L", type="T")
    assert use_kg.format(language="L", type="T")
