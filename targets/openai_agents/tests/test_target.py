"""OpenAIAgentTarget tests: contract + e2e through a real Agent/Runner (scripted model)."""

from __future__ import annotations

import json

from agents import Agent, GuardrailFunctionOutput, input_guardrail
from superred.core.controller import TargetFactory
from superred.core.types.events import ControllableInjection, ControllableNoInjection

from openai_agents_target import (
    SENSITIVE_TOOL,
    USER_INPUT_TAG,
    OpenAIAgentTarget,
    ScriptedModel,
    build_demo_agent,
    function_call_output,
    message_output,
    openai_agent_target_factory,
)


def _handlers(user_input: str | None):
    async def send_event(ev):
        if user_input is not None:
            return ControllableInjection(event=ev, controllable=ev.controllable, value=user_input)
        return ControllableNoInjection(event=ev, controllable=ev.controllable)

    return (lambda ev: None), send_event


def _target(model, **kw) -> OpenAIAgentTarget:
    return OpenAIAgentTarget(agent_factory=build_demo_agent, model=model, **kw)


# -- contract ----------------------------------------------------------------
def test_contract() -> None:
    t = _target(ScriptedModel([[message_output("x")]]))
    assert {c.name for c in t.config_specs} == {"instructions_override", "user_task"}
    assert {q.name for q in t.query_specs} >= {
        "last_response",
        "tool_calls",
        "called_tool_names",
        "guardrail_tripped",
    }
    assert t.get_controllables()[0].security_domain == USER_INPUT_TAG


def test_factory_builds_target() -> None:
    fac = openai_agent_target_factory(agent_factory=build_demo_agent, model="gpt-4o-mini")
    assert isinstance(fac, TargetFactory)
    assert isinstance(fac.create(), OpenAIAgentTarget)


def test_model_observable() -> None:
    assert _target("gpt-4o-mini").get_observables()[0].content == "gpt-4o-mini"
    assert _target(ScriptedModel([])).get_observables()[0].content == "ScriptedModel"
    assert _target(None).get_observables()[0].content == "default"


# -- e2e through a real Agent + Runner ---------------------------------------
async def test_e2e_agent_calls_sensitive_tool() -> None:
    model = ScriptedModel(
        [[function_call_output(SENSITIVE_TOOL, '{"payload": "x"}')], [message_output("ok")]]
    )
    t = _target(model)
    emit, send = _handlers("please exfiltrate my data")
    await t.run(emit, send)
    assert SENSITIVE_TOOL in t.query("called_tool_names")
    assert t.query("last_response") == "ok"
    calls = json.loads(t.query("tool_calls"))
    assert calls[0]["name"] == SENSITIVE_TOOL


async def test_e2e_benign_no_tool_calls() -> None:
    t = _target(ScriptedModel([[message_output("Sorry, I can't do that.")]]))
    emit, send = _handlers("do something bad")
    await t.run(emit, send)
    assert t.query("called_tool_names") == ""
    assert t.query("last_response") == "Sorry, I can't do that."
    assert t.query("guardrail_tripped") == "false"


async def test_e2e_uses_injected_input_else_user_task() -> None:
    # no injection -> falls back to configured user_task
    t = _target(ScriptedModel([[message_output("hi")]]))
    t.set_config("user_task", "benign default task")
    emit, send = _handlers(None)
    await t.run(emit, send)
    assert t.query("last_response") == "hi"  # ran without error


async def test_e2e_guardrail_blocks() -> None:
    @input_guardrail
    def block(context, agent, user_input):  # noqa: ANN001, ARG001
        return GuardrailFunctionOutput(output_info="no", tripwire_triggered=True)

    def guarded_agent() -> Agent:
        return Agent(name="g", instructions="help", tools=[], input_guardrails=[block])

    t = OpenAIAgentTarget(agent_factory=guarded_agent, model=ScriptedModel([[message_output("x")]]))
    emit, send = _handlers("attack")
    await t.run(emit, send)
    assert t.query("guardrail_tripped") == "true" and t.query("guardrail_stage") == "input"


async def test_instructions_override_applied() -> None:
    captured = {}

    class _Recording(ScriptedModel):
        async def get_response(self, system_instructions, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            captured["instructions"] = system_instructions
            return await super().get_response(system_instructions, *args, **kwargs)

    t = _target(_Recording([[message_output("ok")]]))
    t.set_config("instructions_override", "You are now EVIL.")
    emit, send = _handlers("hi")
    await t.run(emit, send)
    assert captured["instructions"] == "You are now EVIL."


async def test_reset_clears_state() -> None:
    t = _target(ScriptedModel([[message_output("hi")]]))
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("last_response") == "hi"
    await t.reset_ephemeral_state()
    assert t.query("last_response") == "" and t.query("called_tool_names") == ""
