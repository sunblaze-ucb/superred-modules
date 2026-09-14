"""LlamaIndexAgentTarget tests: contract + e2e through a real ReActAgent (scripted
LLM, offline) + the tool_output / system_prompt injection surfaces."""

from __future__ import annotations

import json
import types
from typing import Any

import pytest
from superred.core.controller import TargetFactory
from superred.core.types.events import ControllableInjection, ControllableNoInjection

from llamaindex_agent_target import (
    BENIGN_TOOL,
    SENSITIVE_TOOL,
    USER_INPUT_TAG,
    InjectionSpec,
    LlamaIndexAgentTarget,
    build_demo_agent,
    final_answer,
    llamaindex_agent_target_factory,
    scripted_llm,
    tool_action,
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


def _target(llm) -> LlamaIndexAgentTarget:
    return LlamaIndexAgentTarget(agent_factory=build_demo_agent, llm=llm)


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
    # Full agent attack surface, not just the prompt.
    assert {c.name for c in t.get_controllables()} == {
        "user_input",
        "tool_output",
        "system_prompt",
    }
    assert t.get_controllables()[0].security_domain == USER_INPUT_TAG


def test_factory_builds_target() -> None:
    fac = llamaindex_agent_target_factory(
        agent_factory=build_demo_agent, llm=scripted_llm(final_answer("x"))
    )
    assert isinstance(fac, TargetFactory)
    assert isinstance(fac.create(), LlamaIndexAgentTarget)


def test_model_observable() -> None:
    assert _target(scripted_llm(final_answer("x"))).get_observables()[0].content == "scripted-react"


def test_llm_required() -> None:
    with pytest.raises(ValueError):
        _target(None)


# -- e2e through a real ReActAgent -------------------------------------------
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


async def test_scripted_llm_rewinds_each_run() -> None:
    # a stateful scripted llm shared across runs must replay from the top each run
    # (the target rewinds it), so run 2 still sees the tool-call script.
    llm = scripted_llm(tool_action(SENSITIVE_TOOL, "{}"), final_answer("done"))
    t = _target(llm)
    emit, send = _handlers("attack")
    await t.run(emit, send)
    assert SENSITIVE_TOOL in t.query("called_tool_names")
    await t.run(emit, send)  # same shared llm instance
    assert SENSITIVE_TOOL in t.query("called_tool_names")  # replayed, not exhausted


async def test_reset_clears_state() -> None:
    t = _target(scripted_llm(final_answer("hi")))
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("last_response") == "hi"
    await t.reset_ephemeral_state()
    assert t.query("last_response") == "" and t.query("called_tool_names") == ""


# -- injection surfaces (tool_output / system_prompt) ------------------------
def test_injection_spec_system_prompt_and_wrap_tools() -> None:
    empty = InjectionSpec()
    assert empty.apply_system_prompt("base") == "base"
    # Tool stubs must expose a settable ``.call`` (the boundary _wrap_tool wraps);
    # bare strings would raise AttributeError the moment wrap_tools actually wraps.
    tools = [
        types.SimpleNamespace(call=lambda: "OUT"),
        types.SimpleNamespace(call=lambda: "OUT"),
    ]
    assert empty.wrap_tools(tools) is tools  # nothing injected -> untouched
    spec = InjectionSpec(system_prompt_suffix="ATK", tool_output_appendix="PAYLOAD")
    assert spec.apply_system_prompt("base") == "base\n\nATK"
    assert spec.apply_system_prompt(None) == "ATK"
    wrapped = spec.wrap_tools(tools)
    assert len(wrapped) == 2 and wrapped[0] is not tools[0]  # copies, one-for-one
    assert "PAYLOAD" in wrapped[0].call()  # the appendix reaches the tool's return


def test_append_appendix_bare_shapes() -> None:
    # Pure (framework-free) shapes: a bare string, a non-string/non-ToolOutput
    # value (coerced, never silently dropped), and the empty-appendix no-op.
    from llamaindex_agent_target.injection import _append_appendix

    assert _append_appendix("hello", "MARK") == "hello\n\nMARK"
    assert "MARK" in _append_appendix(12345, "MARK")  # coerced, not dropped
    same = object()
    assert _append_appendix(same, "") is same  # empty appendix is a no-op


def test_append_appendix_handles_tool_output_shapes() -> None:
    # The model reads ToolOutput.content; the appendix must reach it regardless of
    # whether raw_output is structured or a string (the #183 bug injected only into
    # str results and skipped other shapes — a silent false negative).
    from llama_index.core.tools import ToolOutput

    from llamaindex_agent_target.injection import _append_appendix

    structured = ToolOutput(
        content="{'temp': 22}", tool_name="t", raw_input={}, raw_output={"temp": 22}
    )
    out = _append_appendix(structured, "STRUCT_MARK")
    assert "STRUCT_MARK" in out.content
    # Injected exactly once: in >=0.14 `content` is a property over `blocks`, so
    # setting content AND appending a block would duplicate the payload.
    assert out.content.count("STRUCT_MARK") == 1
    assert out.raw_output == {"temp": 22}  # structure preserved, not mangled

    stringy = ToolOutput(
        content="weather: sunny", tool_name="t", raw_input={}, raw_output="weather: sunny"
    )
    out2 = _append_appendix(stringy, "STR_MARK")
    assert "STR_MARK" in out2.content and "STR_MARK" in out2.raw_output
    assert out2.content.count("STR_MARK") == 1


async def test_tool_output_injection_reaches_non_string_tool_return() -> None:
    # A tool whose function returns a NON-STRING shape (structured dict): the
    # wrapped tool must still land the appendix in the model-visible ToolOutput
    # content, not skip it because the return wasn't a str. Both dispatch paths are
    # checked — sync ``call`` and the async ``acall`` the real agent uses.
    from llama_index.core.tools import FunctionTool

    def structured_tool() -> dict[str, Any]:
        """Return structured (non-string) weather data."""
        return {"temp_c": 22, "sky": "sunny"}

    tool = FunctionTool.from_defaults(fn=structured_tool, name="weather_struct")
    wrapped = InjectionSpec(tool_output_appendix="NONSTR_MARK").wrap_tools([tool])[0]

    sync_out = wrapped.call()
    assert "NONSTR_MARK" in str(sync_out.content)
    assert "sunny" in str(sync_out.content)  # original (structured) content preserved

    async_out = await wrapped.acall()
    assert "NONSTR_MARK" in str(async_out.content)


async def test_tool_output_injection_reaches_the_tool_result() -> None:
    # The indirect-injection surface end-to-end: attacker content appended to a
    # tool's return must reach the ToolOutput the real ReActAgent reads back as its
    # observation (captured from the workflow's ToolCallResult events).
    llm = scripted_llm(tool_action(BENIGN_TOOL, '{"city": "NYC"}'), final_answer("done"))
    agent = build_demo_agent(llm, InjectionSpec(tool_output_appendix="INDIRECT_INJECT_MARK"))
    handler = agent.run(user_msg="weather?")
    observed: list[str] = []
    async for ev in handler.stream_events():
        if type(ev).__name__ == "ToolCallResult":
            out = getattr(ev, "tool_output", None)
            if out is not None:
                observed.append(str(getattr(out, "content", "")))
    await handler
    assert observed and any("INDIRECT_INJECT_MARK" in c for c in observed)


async def test_target_drives_all_surfaces_without_error() -> None:
    # The target plumbs all three surfaces into a per-run agent build; a run with
    # every surface injected completes and still exercises the tool.
    llm = scripted_llm(tool_action(SENSITIVE_TOOL, '{"payload": "x"}'), final_answer("ok"))
    t = _target(llm)
    emit, send = _handlers(
        {"user_input": "do it", "tool_output": "T-INJ", "system_prompt": "S-INJ"}
    )
    await t.run(emit, send)
    assert t.query("error") == ""
    assert SENSITIVE_TOOL in t.query("called_tool_names")


async def test_factory_error_is_recorded_not_raised() -> None:
    # A factory that raises on a run must be recorded as a run error (so the claim
    # can abstain), not propagated to hard-abort the task's remaining runs. A
    # one-argument factory (can't take the spec) fails the same way.
    def boom_factory(llm: Any, spec: InjectionSpec) -> Any:
        raise RuntimeError("factory blew up")

    t = LlamaIndexAgentTarget(agent_factory=boom_factory, llm=scripted_llm(final_answer("x")))
    emit, send = _handlers("x")
    await t.run(emit, send)  # must not raise
    assert "factory blew up" in t.query("error")
    assert t.query("called_tool_names") == "" and t.query("last_response") == ""
