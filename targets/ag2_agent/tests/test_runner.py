"""run_agent_capture tests: a real ag2.Agent driven by a scripted TestConfig, plus
error capture. Offline (no network)."""

from __future__ import annotations

from ag2_agent_target import (
    SENSITIVE_TOOL,
    build_demo_agent,
    message_turn,
    run_agent_capture,
    scripted_config,
    tool_call_turn,
)


async def test_message_only_run() -> None:
    agent = build_demo_agent(scripted_config(message_turn("hello there")))
    result = await run_agent_capture(agent=agent, user_input="hi")
    assert result.final_response == "hello there"
    assert result.called_tool_names == [] and result.error == ""


async def test_tool_call_then_answer() -> None:
    cfg = scripted_config(
        tool_call_turn(SENSITIVE_TOOL, '{"payload": "secret"}'),
        message_turn("done"),
    )
    agent = build_demo_agent(cfg)
    result = await run_agent_capture(agent=agent, user_input="hi")
    assert SENSITIVE_TOOL in result.called_tool_names
    assert result.final_response == "done"
    assert "secret" in result.tool_calls[0].arguments  # arguments captured


async def test_run_error_is_captured() -> None:
    # a scripted BaseException turn makes the model "raise"; the error is recorded,
    # never propagated out of the runner.
    agent = build_demo_agent(scripted_config(RuntimeError("model down")))
    result = await run_agent_capture(agent=agent, user_input="hi")
    assert result.error != "" and result.final_response == ""
