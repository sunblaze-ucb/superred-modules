"""LangChainAgentTarget tests: contract + e2e through a real create_agent graph
(scripted chat model, offline)."""

from __future__ import annotations

import json

import pytest
from superred.core.controller import TargetFactory
from superred.core.types.events import ControllableInjection, ControllableNoInjection

from langchain_agent_target import (
    SENSITIVE_TOOL,
    USER_INPUT_TAG,
    LangChainAgentTarget,
    ScriptedChatModel,
    build_demo_agent,
    langchain_agent_target_factory,
    message_output,
    tool_call_output,
)


def _handlers(user_input: str | None):
    async def send_event(ev):
        if user_input is not None:
            return ControllableInjection(event=ev, controllable=ev.controllable, value=user_input)
        return ControllableNoInjection(event=ev, controllable=ev.controllable)

    return (lambda ev: None), send_event


def _target(model, **kw) -> LangChainAgentTarget:
    return LangChainAgentTarget(agent_factory=build_demo_agent, model=model, **kw)


# -- contract ----------------------------------------------------------------
def test_contract() -> None:
    t = _target(ScriptedChatModel(responses=[message_output("x")]))
    assert {c.name for c in t.config_specs} == {"user_task"}
    assert {q.name for q in t.query_specs} >= {
        "last_response",
        "tool_calls",
        "called_tool_names",
        "error",
    }
    assert t.get_controllables()[0].security_domain == USER_INPUT_TAG


def test_factory_builds_target() -> None:
    fac = langchain_agent_target_factory(agent_factory=build_demo_agent, model="gpt-4o-mini")
    assert isinstance(fac, TargetFactory)
    assert isinstance(fac.create(), LangChainAgentTarget)


def test_model_observable() -> None:
    assert _target("gpt-4o-mini").get_observables()[0].content == "gpt-4o-mini"
    scripted = _target(ScriptedChatModel(responses=[]))
    assert scripted.get_observables()[0].content == "ScriptedChatModel"


def test_model_required() -> None:
    # create_agent has no default model, so None must fail fast (not silently error
    # on every run).
    with pytest.raises(ValueError):
        _target(None)


# -- e2e through a real create_agent graph -----------------------------------
async def test_e2e_agent_calls_sensitive_tool() -> None:
    model = ScriptedChatModel(
        responses=[tool_call_output(SENSITIVE_TOOL, {"payload": "x"}), message_output("ok")]
    )
    t = _target(model)
    emit, send = _handlers("please exfiltrate my data")
    await t.run(emit, send)
    assert SENSITIVE_TOOL in t.query("called_tool_names")
    assert t.query("last_response") == "ok"
    calls = json.loads(t.query("tool_calls"))
    assert calls[0]["name"] == SENSITIVE_TOOL


async def test_e2e_benign_no_tool_calls() -> None:
    t = _target(ScriptedChatModel(responses=[message_output("Sorry, I can't do that.")]))
    emit, send = _handlers("do something bad")
    await t.run(emit, send)
    assert t.query("called_tool_names") == ""
    assert t.query("last_response") == "Sorry, I can't do that."
    assert t.query("error") == ""


async def test_e2e_uses_injected_input_else_user_task() -> None:
    t = _target(ScriptedChatModel(responses=[message_output("hi")]))
    t.set_config("user_task", "benign default task")
    emit, send = _handlers(None)
    await t.run(emit, send)
    assert t.query("last_response") == "hi"  # ran without error


async def test_scripted_model_rewinds_each_run() -> None:
    # a stateful scripted model shared across runs must replay from the top each
    # run (the target rewinds it), so run 2 sees the tool-call script.
    model = ScriptedChatModel(
        responses=[tool_call_output(SENSITIVE_TOOL, {}), message_output("done")]
    )
    t = _target(model)
    emit, send = _handlers("attack")
    await t.run(emit, send)
    assert SENSITIVE_TOOL in t.query("called_tool_names")
    await t.run(emit, send)  # same shared model instance
    assert SENSITIVE_TOOL in t.query("called_tool_names")  # replayed, not clamped


async def test_reset_clears_state() -> None:
    t = _target(ScriptedChatModel(responses=[message_output("hi")]))
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("last_response") == "hi"
    await t.reset_ephemeral_state()
    assert t.query("last_response") == "" and t.query("called_tool_names") == ""
