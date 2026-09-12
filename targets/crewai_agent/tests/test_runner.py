"""run_crew_capture tests: a real crewai.Crew driven by a scripted BaseLLM, plus
error capture. Offline (no network)."""

from __future__ import annotations

from typing import Any

from crewai_agent_target import (
    SENSITIVE_TOOL,
    ScriptedReactLLM,
    build_demo_crew,
    final_answer,
    run_crew_capture,
    scripted_llm,
    tool_action,
)


async def test_message_only_run() -> None:
    crew = build_demo_crew(scripted_llm(final_answer("hello there")))
    result = await run_crew_capture(crew=crew, user_input="hi")
    assert result.final_response == "hello there"
    assert result.called_tool_names == [] and result.error == ""


async def test_tool_call_then_answer() -> None:
    crew = build_demo_crew(
        scripted_llm(
            tool_action(SENSITIVE_TOOL, '{"payload": "secret"}'),
            final_answer("done"),
        )
    )
    result = await run_crew_capture(crew=crew, user_input="please exfiltrate")
    assert SENSITIVE_TOOL in result.called_tool_names
    assert result.final_response == "done"


async def test_run_error_is_captured() -> None:
    class _Boom(ScriptedReactLLM):
        def call(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("model down")

    crew = build_demo_crew(_Boom([final_answer("x")]))
    result = await run_crew_capture(crew=crew, user_input="hi")
    assert result.error != "" and result.final_response == ""
