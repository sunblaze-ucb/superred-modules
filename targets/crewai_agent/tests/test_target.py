"""CrewAIAgentTarget tests: contract + e2e through a real crewai.Crew (scripted LLM, offline).

These exercise a real ``crewai.Crew`` and so require ``crewai`` installed (CI); the
pure ``InjectionSpec`` shape logic is covered offline in ``test_injection.py``.
"""

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
    # Full agent attack surface, not just the prompt.
    assert {c.name for c in t.get_controllables()} == {
        "user_input",
        "tool_output",
        "system_prompt",
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


async def test_captures_tools_even_with_agent_step_callback() -> None:
    # a crew_factory whose Agent already has its own step_callback must not silently
    # defeat tool capture (CrewAI keeps the agent's callback over the crew's).
    def factory_with_agent_cb(llm, injection):
        crew = build_demo_crew(llm, injection)
        crew.agents[0].step_callback = lambda step: None  # pre-existing agent callback
        return crew

    t = CrewAIAgentTarget(
        crew_factory=factory_with_agent_cb,
        llm=scripted_llm(tool_action(SENSITIVE_TOOL, "{}"), final_answer("ok")),
    )
    emit, send = _handlers("attack")
    await t.run(emit, send)
    assert SENSITIVE_TOOL in t.query("called_tool_names")


async def test_captures_tools_even_if_agent_callback_raises() -> None:
    # a foreign agent step_callback that raises must not drop our tool capture
    # (we record before invoking it).
    def factory_with_raising_cb(llm, injection):
        crew = build_demo_crew(llm, injection)

        def boom(step):  # noqa: ANN001, ANN202
            raise RuntimeError("foreign callback boom")

        crew.agents[0].step_callback = boom
        return crew

    t = CrewAIAgentTarget(
        crew_factory=factory_with_raising_cb,
        llm=scripted_llm(tool_action(SENSITIVE_TOOL, "{}"), final_answer("ok")),
    )
    emit, send = _handlers("attack")
    await t.run(emit, send)
    assert SENSITIVE_TOOL in t.query("called_tool_names")


async def test_captures_and_preserves_crew_level_callback() -> None:
    # a crew-level step_callback (Crew(step_callback=...)) must still fire even though
    # we set agent-level recorders (which stops CrewAI copying the crew callback).
    seen = []

    def factory_with_crew_cb(llm, injection):
        crew = build_demo_crew(llm, injection)
        crew.step_callback = lambda step: seen.append(1)
        return crew

    t = CrewAIAgentTarget(
        crew_factory=factory_with_crew_cb,
        llm=scripted_llm(tool_action(SENSITIVE_TOOL, "{}"), final_answer("ok")),
    )
    emit, send = _handlers("attack")
    await t.run(emit, send)
    assert SENSITIVE_TOOL in t.query("called_tool_names")  # captured
    assert seen  # the caller's crew-level callback still fired


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


async def test_crew_factory_error_recorded_not_raised() -> None:
    # A factory that raises on a run must be recorded as a run error (so the claim
    # can abstain), not propagated to hard-abort the task's remaining runs.
    def bad_factory(llm, spec):
        raise ValueError("bad crew config")

    t = CrewAIAgentTarget(crew_factory=bad_factory, llm=scripted_llm(final_answer("x")))
    emit, send = _handlers("q")
    await t.run(emit, send)  # must not raise
    assert "bad crew config" in t.query("error")
    assert t.query("called_tool_names") == "" and t.query("last_response") == ""


async def test_reset_clears_state() -> None:
    t = _target(scripted_llm(final_answer("hi")))
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("last_response") == "hi"
    await t.reset_ephemeral_state()
    assert t.query("last_response") == "" and t.query("called_tool_names") == ""


# -- injection surfaces (tool_output / system_prompt) ------------------------
async def test_target_drives_all_surfaces_without_error() -> None:
    # The target plumbs all three surfaces into a per-run crew build; a run with
    # every surface injected completes and still exercises the tool.
    llm = scripted_llm(tool_action(SENSITIVE_TOOL, '{"payload": "x"}'), final_answer("ok"))
    t = _target(llm)
    emit, send = _handlers(
        {"user_input": "do it", "tool_output": "T-INJ", "system_prompt": "S-INJ"}
    )
    await t.run(emit, send)
    assert t.query("error") == ""
    assert SENSITIVE_TOOL in t.query("called_tool_names")
