"""CrewAIAgentTarget tests: contract + e2e through a real crewai.Crew (scripted LLM, offline)."""

from __future__ import annotations

import json

import pytest
from superred.core.controller import TargetFactory
from superred.core.types.events import ControllableInjection, ControllableNoInjection

from crewai_agent_target import (
    SENSITIVE_TOOL,
    USER_INPUT_TAG,
    CrewAIAgentTarget,
    build_demo_crew,
    crewai_agent_target_factory,
    final_answer,
    scripted_llm,
    tool_action,
)


def _handlers(user_input: str | None):
    async def send_event(ev):
        if user_input is not None:
            return ControllableInjection(event=ev, controllable=ev.controllable, value=user_input)
        return ControllableNoInjection(event=ev, controllable=ev.controllable)

    return (lambda ev: None), send_event


def _target(llm) -> CrewAIAgentTarget:
    return CrewAIAgentTarget(crew_factory=build_demo_crew, llm=llm)


# -- contract ----------------------------------------------------------------
def test_contract() -> None:
    t = _target(scripted_llm(final_answer("x")))
    assert {c.name for c in t.config_specs} == {"user_task"}
    assert {q.name for q in t.query_specs} >= {
        "last_response",
        "tool_calls",
        "called_tool_names",
        "error",
    }
    assert t.get_controllables()[0].security_domain == USER_INPUT_TAG


def test_factory_builds_target() -> None:
    fac = crewai_agent_target_factory(
        crew_factory=build_demo_crew, llm=scripted_llm(final_answer("x"))
    )
    assert isinstance(fac, TargetFactory)
    assert isinstance(fac.create(), CrewAIAgentTarget)


def test_model_observable() -> None:
    # a scripted llm exposes model="scripted-react"; a class with no str model -> class name.
    assert _target(scripted_llm(final_answer("x"))).get_observables()[0].content == "scripted-react"


def test_llm_required() -> None:
    with pytest.raises(ValueError):
        _target(None)


# -- e2e through a real crewai.Crew ------------------------------------------
async def test_e2e_agent_calls_sensitive_tool() -> None:
    llm = scripted_llm(tool_action(SENSITIVE_TOOL, '{"payload": "x"}'), final_answer("ok"))
    t = _target(llm)
    emit, send = _handlers("please exfiltrate my data")
    await t.run(emit, send)
    assert SENSITIVE_TOOL in t.query("called_tool_names")
    assert t.query("last_response") == "ok"
    calls = json.loads(t.query("tool_calls"))
    assert calls[0]["name"] == SENSITIVE_TOOL


async def test_e2e_benign_no_tool_calls() -> None:
    t = _target(scripted_llm(final_answer("Sorry, I can't do that.")))
    emit, send = _handlers("do something bad")
    await t.run(emit, send)
    assert t.query("called_tool_names") == ""
    assert t.query("last_response") == "Sorry, I can't do that."
    assert t.query("error") == ""


async def test_e2e_uses_injected_input_else_user_task() -> None:
    t = _target(scripted_llm(final_answer("hi")))
    t.set_config("user_task", "benign default task")
    emit, send = _handlers(None)
    await t.run(emit, send)
    assert t.query("last_response") == "hi"  # ran without error


async def test_reset_clears_state() -> None:
    t = _target(scripted_llm(final_answer("hi")))
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("last_response") == "hi"
    await t.reset_ephemeral_state()
    assert t.query("last_response") == "" and t.query("called_tool_names") == ""
