"""run_agent_capture tests: real create_agent graph driven by a scripted chat
model, plus error capture. Offline (no network)."""

from __future__ import annotations

from typing import Any

from langchain_agent_target import (
    SENSITIVE_TOOL,
    ScriptedChatModel,
    build_demo_agent,
    message_output,
    run_agent_capture,
    tool_call_output,
)


async def test_message_only_run() -> None:
    agent = build_demo_agent(ScriptedChatModel(responses=[message_output("hello there")]))
    result = await run_agent_capture(agent=agent, user_input="hi")
    assert result.final_response == "hello there"
    assert result.called_tool_names == [] and result.error == ""


async def test_tool_call_then_answer() -> None:
    model = ScriptedChatModel(
        responses=[
            tool_call_output(SENSITIVE_TOOL, {"payload": "secret"}),
            message_output("done"),
        ]
    )
    agent = build_demo_agent(model)
    result = await run_agent_capture(agent=agent, user_input="hi")
    assert SENSITIVE_TOOL in result.called_tool_names
    assert result.final_response == "done"
    assert "secret" in result.tool_calls[0].arguments  # arguments captured


async def test_tool_loop_hits_recursion_limit_cleanly() -> None:
    # a script whose (repeating) last element is a tool call must not corrupt the
    # graph's id-keyed message state: the run stops at the recursion limit and the
    # error is recorded cleanly (not an opaque KeyError from duplicate message ids).
    model = ScriptedChatModel(responses=[tool_call_output(SENSITIVE_TOOL, {})])
    agent = build_demo_agent(model)
    result = await run_agent_capture(agent=agent, user_input="loop", recursion_limit=4)
    assert result.error != ""  # recorded, never raised
    assert "KeyError" not in result.error  # not the duplicate-id corruption
    assert "recursion" in result.error.lower()  # a clean recursion-limit stop
    # ainvoke raised before returning, so no messages are available to extract.
    assert result.called_tool_names == [] and result.final_response == ""


async def test_run_error_is_captured() -> None:
    class _Boom(ScriptedChatModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model down")

    agent = build_demo_agent(_Boom(responses=[message_output("x")]))
    result = await run_agent_capture(agent=agent, user_input="hi")
    assert "model down" in result.error and result.final_response == ""
