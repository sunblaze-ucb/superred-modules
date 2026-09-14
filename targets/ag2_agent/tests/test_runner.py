"""run_agent_capture tests: a real ag2.Agent driven by a scripted TestConfig, plus
error capture. Offline (no network)."""

from __future__ import annotations

from typing import Any

from ag2.events.tool_events import ToolCallEvent

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


async def test_salvaged_tool_calls_survive_aexit_error() -> None:
    # Compound failure: a tool call happened, then run.result() raised, AND the
    # run's __aexit__ raises during teardown. The salvaged tool call must NOT be
    # dropped (it is the claim-scoring surface) — regression test for the outer
    # except early-return.
    class _History:
        def __init__(self, events: list[Any]) -> None:
            self._events = events

        async def get_events(self) -> list[Any]:
            return self._events

    class _Stream:
        def __init__(self, events: list[Any]) -> None:
            self.history = _History(events)

    class _Run:
        def __init__(self, events: list[Any]) -> None:
            self.stream = _Stream(events)

        async def __aenter__(self) -> _Run:
            return self

        async def __aexit__(self, *exc: object) -> bool:
            raise RuntimeError("aexit boom during teardown")

        async def result(self) -> Any:
            raise RuntimeError("model down mid-run")

    class _Agent:
        def __init__(self, events: list[Any]) -> None:
            self._events = events

        def run(self, user_input: str) -> _Run:
            return _Run(self._events)

    events = [ToolCallEvent(name=SENSITIVE_TOOL, arguments='{"payload": "x"}')]
    result = await run_agent_capture(agent=_Agent(events), user_input="attack")
    assert result.error != ""  # an error was recorded
    assert SENSITIVE_TOOL in result.called_tool_names  # but the tool call survived
