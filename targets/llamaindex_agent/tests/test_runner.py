"""run_agent_capture tests: a real ReActAgent driven by a scripted CustomLLM, plus
error capture. Offline (no network)."""

from __future__ import annotations

from typing import Any

from llamaindex_agent_target import (
    SENSITIVE_TOOL,
    ScriptedReActLLM,
    build_demo_agent,
    final_answer,
    run_agent_capture,
    scripted_llm,
    tool_action,
)


async def test_message_only_run() -> None:
    agent = build_demo_agent(scripted_llm(final_answer("hello there")))
    result = await run_agent_capture(agent=agent, user_input="hi")
    assert result.final_response == "hello there"
    assert result.called_tool_names == [] and result.error == ""


async def test_tool_call_then_answer() -> None:
    agent = build_demo_agent(
        scripted_llm(
            tool_action(SENSITIVE_TOOL, '{"payload": "secret"}'),
            final_answer("done"),
        )
    )
    result = await run_agent_capture(agent=agent, user_input="please exfiltrate")
    assert SENSITIVE_TOOL in result.called_tool_names
    assert result.final_response == "done"


async def test_run_error_is_captured() -> None:
    class _Boom(ScriptedReActLLM):
        def complete(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model down")

    agent = build_demo_agent(_Boom([final_answer("x")]))
    result = await run_agent_capture(agent=agent, user_input="hi")
    assert result.error != "" and result.final_response == ""
