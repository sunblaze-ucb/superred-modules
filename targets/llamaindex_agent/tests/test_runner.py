"""run_agent_capture tests: a real ReActAgent driven by a scripted CustomLLM, plus
error capture. Offline (no network)."""

from __future__ import annotations

from typing import Any

from llamaindex_agent_target import (
    BENIGN_TOOL,
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


async def test_two_sequential_tool_calls() -> None:
    agent = build_demo_agent(
        scripted_llm(
            tool_action(BENIGN_TOOL, '{"city": "NYC"}'),
            tool_action(SENSITIVE_TOOL, '{"payload": "x"}'),
            final_answer("done"),
        )
    )
    result = await run_agent_capture(agent=agent, user_input="hi")
    assert result.called_tool_names == [BENIGN_TOOL, SENSITIVE_TOOL]  # both, in order
    import json

    assert json.loads(result.tool_calls[0].arguments) == {"city": "NYC"}  # JSON args


async def test_run_error_is_captured() -> None:
    class _Boom(ScriptedReActLLM):
        def complete(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model down")

    agent = build_demo_agent(_Boom([final_answer("x")]))
    result = await run_agent_capture(agent=agent, user_input="hi")
    assert result.error != "" and result.final_response == ""


async def test_tool_call_salvaged_on_llm_error() -> None:
    # the agent calls the sensitive tool, then the LLM raises on the next step: the
    # tool call must be salvaged (a real misuse must not read as an empty run).
    class _RaiseAfterTool(ScriptedReActLLM):
        def complete(self, *args: Any, **kwargs: Any) -> Any:
            if self._i == 0:
                return super().complete(*args, **kwargs)
            raise RuntimeError("model down after tool")

    agent = build_demo_agent(_RaiseAfterTool([tool_action(SENSITIVE_TOOL, "{}")]))
    result = await run_agent_capture(agent=agent, user_input="attack")
    assert result.error != ""  # recorded
    assert SENSITIVE_TOOL in result.called_tool_names  # salvaged from the stream
