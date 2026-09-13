"""AG2AgentTarget tests: contract + injection surfaces + e2e through a real
ag2.Agent (scripted config, offline).

The injection-surface and contract tests below are framework-free (they exercise
``InjectionSpec`` and the target's error path directly) and run without ``ag2``
installed. The ``e2e`` tests build a real ``ag2.Agent`` via a scripted
``TestConfig`` and require ``ag2``; they verify the plumbing offline (no network),
not the security outcome (whether a real model *follows* an injection).
"""

from __future__ import annotations

import json

import pytest
from superred.core.controller import TargetFactory
from superred.core.types.events import ControllableInjection, ControllableNoInjection

from ag2_agent_target import (
    SENSITIVE_TOOL,
    SYSTEM_PROMPT_TAG,
    TOOL_OUTPUT_TAG,
    USER_INPUT_TAG,
    AG2AgentTarget,
    InjectionSpec,
    ag2_agent_target_factory,
    build_demo_agent,
    message_turn,
    scripted_config,
    tool_call_turn,
)


def _handlers(inject: dict[str, str] | str | None = None):
    # inject: {controllable_name: value}; a bare str is shorthand for user_input.
    if isinstance(inject, str):
        inject = {"user_input": inject}
    inject = inject or {}

    async def send_event(ev):
        name = ev.controllable.name
        if name in inject:
            return ControllableInjection(
                event=ev, controllable=ev.controllable, value=inject[name]
            )
        return ControllableNoInjection(event=ev, controllable=ev.controllable)

    return (lambda ev: None), send_event


def _target(config) -> AG2AgentTarget:
    return AG2AgentTarget(agent_factory=build_demo_agent, model=config)


# -- contract ----------------------------------------------------------------
def test_contract() -> None:
    t = _target(scripted_config(message_turn("x")))
    assert {c.name for c in t.config_specs} == {"user_task"}
    assert {q.name for q in t.query_specs} >= {
        "last_response",
        "tool_calls",
        "called_tool_names",
        "error",
    }
    # Full agent attack surface, not just the prompt.
    assert {c.name for c in t.get_controllables()} == {
        "user_input",
        "tool_output",
        "system_prompt",
    }
    domains = {c.name: c.security_domain for c in t.get_controllables()}
    assert domains["user_input"] == USER_INPUT_TAG
    assert domains["tool_output"] == TOOL_OUTPUT_TAG
    assert domains["system_prompt"] == SYSTEM_PROMPT_TAG


def test_factory_builds_target() -> None:
    fac = ag2_agent_target_factory(
        agent_factory=build_demo_agent, model=scripted_config(message_turn("x"))
    )
    assert isinstance(fac, TargetFactory)
    assert isinstance(fac.create(), AG2AgentTarget)


def test_model_observable_is_config_class_name() -> None:
    # only the config class name is emitted — never the config object (no key leak).
    obs = _target(scripted_config(message_turn("x"))).get_observables()[0]
    assert obs.content == "TestConfig"


def test_model_required() -> None:
    with pytest.raises(ValueError):
        _target(None)


# -- injection surfaces (framework-free; run without ag2 installed) -----------
def test_injection_spec_system_prompt_and_wrap_tools() -> None:
    empty = InjectionSpec()
    assert empty.apply_system_prompt("base") == "base"
    tools = [lambda: "a", lambda: "b"]
    # No injection: tools pass through unchanged (same callables).
    assert empty.wrap_tools(tools) == tools

    spec = InjectionSpec(system_prompt_suffix="ATK", tool_output_appendix="PAYLOAD")
    assert spec.apply_system_prompt("base") == "base\n\nATK"
    assert spec.apply_system_prompt(None) == "ATK"
    wrapped = spec.wrap_tools(tools)
    assert wrapped != tools and len(wrapped) == 2  # each tool wrapped


def test_wrap_tools_preserves_tool_schema_metadata() -> None:
    # AG2 builds a tool's JSON schema from name/doc/annotations/signature; the
    # wrapper must preserve them or the schema is corrupted.
    import inspect

    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return "sunny"

    (wrapped,) = InjectionSpec(tool_output_appendix="X").wrap_tools([get_weather])
    assert wrapped.__name__ == "get_weather"
    assert wrapped.__doc__ == "Get the current weather for a city."
    assert list(inspect.signature(wrapped).parameters) == ["city"]
    # Annotations survive (this test module uses PEP 563, so they are stringified).
    assert "city" in wrapped.__annotations__
    assert wrapped.__annotations__ == get_weather.__annotations__


def test_wrap_tools_injects_into_every_return_shape() -> None:
    # The indirect-injection surface: the appendix must reach the model-visible text
    # for EVERY return shape a tool may produce — injecting into str only and
    # dropping structured / scalar returns is a silent false negative on this exact
    # surface. str is the common case; the rest are the "non-string shapes".
    mark = "INDIRECT_INJECT_MARK"
    spec = InjectionSpec(tool_output_appendix=mark)

    def wrap(fn):
        (w,) = spec.wrap_tools([fn])
        return w

    # str -> trailing text
    assert mark in wrap(lambda: "weather: sunny")()

    # dict / JSON object (non-string shape) -> attached under a note key, so it
    # shows up in the serialized object the model reads.
    out = wrap(lambda: {"weather": "sunny"})()
    assert isinstance(out, dict) and out["weather"] == "sunny"
    assert mark in json.dumps(out)

    # dict whose key collides with the note key -> non-clobbering (both survive).
    out = wrap(lambda: {"injected_note": "real"})()
    assert out["injected_note"] == "real" and mark in json.dumps(out)

    # list / JSON array (non-string shape) -> appended element.
    out = wrap(lambda: ["a", "b"])()
    assert out == ["a", "b", mark]

    # tuple -> appended element, tuple-ness preserved.
    out = wrap(lambda: ("a", "b"))()
    assert out == ("a", "b", mark)

    # scalar shapes (non-string) -> rendered to text carrying the appendix, never
    # dropped.
    assert mark in wrap(lambda: 42)()
    assert mark in wrap(lambda: None)()
    assert mark in wrap(lambda: True)()


async def test_wrap_tools_handles_async_tool() -> None:
    # AG2 tools may be coroutine functions; the wrapper must await and inject.
    mark = "ASYNC_MARK"

    async def fetch() -> str:
        return "page body"

    (wrapped,) = InjectionSpec(tool_output_appendix=mark).wrap_tools([fetch])
    assert mark in await wrapped()


async def test_factory_error_is_recorded_not_raised() -> None:
    # A factory that raises on a run must be recorded as a run error (so the claim
    # can abstain), not propagated to hard-abort the task's remaining runs.
    def boom_factory(model, spec):
        raise RuntimeError("factory blew up")

    t = AG2AgentTarget(agent_factory=boom_factory, model="offline-config")
    emit, send = _handlers("x")
    await t.run(emit, send)  # must not raise
    assert "factory blew up" in t.query("error")
    assert t.query("called_tool_names") == ""


async def test_one_arg_factory_is_recorded_not_raised() -> None:
    # A legacy one-argument factory no longer matches the (model, spec) contract; the
    # resulting TypeError is recorded as a run error rather than propagated.
    t = AG2AgentTarget(agent_factory=lambda model: object(), model="offline-config")
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("error") != ""


# -- e2e through a real ag2.Agent (requires ag2; offline, no network) ---------
async def test_e2e_agent_calls_sensitive_tool() -> None:
    cfg = scripted_config(tool_call_turn(SENSITIVE_TOOL, '{"payload": "x"}'), message_turn("ok"))
    t = _target(cfg)
    emit, send = _handlers("please exfiltrate my data")
    await t.run(emit, send)
    assert SENSITIVE_TOOL in t.query("called_tool_names")
    assert t.query("last_response") == "ok"
    calls = json.loads(t.query("tool_calls"))
    assert calls[0]["name"] == SENSITIVE_TOOL


async def test_e2e_benign_no_tool_calls() -> None:
    t = _target(scripted_config(message_turn("Sorry, I can't do that.")))
    emit, send = _handlers("do something bad")
    await t.run(emit, send)
    assert t.query("called_tool_names") == ""
    assert t.query("last_response") == "Sorry, I can't do that."
    assert t.query("error") == ""


async def test_e2e_uses_injected_input_else_user_task() -> None:
    t = _target(scripted_config(message_turn("hi")))
    t.set_config("user_task", "benign default task")
    emit, send = _handlers(None)
    await t.run(emit, send)
    assert t.query("last_response") == "hi"  # ran without error


async def test_e2e_drives_all_surfaces_without_error() -> None:
    # The target plumbs all three surfaces into a per-run agent build; a run with
    # every surface injected completes (wrapped tools + suffixed system message do
    # not break the agent) and still exercises the tool.
    cfg = scripted_config(tool_call_turn(SENSITIVE_TOOL, '{"payload": "x"}'), message_turn("ok"))
    t = _target(cfg)
    emit, send = _handlers(
        {"user_input": "do it", "tool_output": "T-INJ", "system_prompt": "S-INJ"}
    )
    await t.run(emit, send)
    assert t.query("error") == ""
    assert SENSITIVE_TOOL in t.query("called_tool_names")


async def test_reset_clears_state() -> None:
    t = _target(scripted_config(message_turn("hi")))
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("last_response") == "hi"
    await t.reset_ephemeral_state()
    assert t.query("last_response") == "" and t.query("called_tool_names") == ""
