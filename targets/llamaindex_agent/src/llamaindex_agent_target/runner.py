"""Run a LlamaIndex ``ReActAgent`` and capture what it did.

Kept separate from the Target so it is unit-testable with a scripted ``CustomLLM``
and a real ``ReActAgent`` — no network. Tool calls are captured by streaming the
agent's workflow events (a ``ToolCall`` event per tool use), so calls made before a
mid-run error are salvaged. Any error is recorded rather than raised, so a claim
can abstain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    """One tool call the agent made."""

    name: str
    arguments: str


@dataclass
class AgentRunResult:
    """The captured outcome of one agent run."""

    final_response: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: str = ""

    @property
    def called_tool_names(self) -> list[str]:
        return [c.name for c in self.tool_calls]


async def run_agent_capture(*, agent: Any, user_input: str) -> AgentRunResult:
    """Run ``agent`` on ``user_input`` and capture its behaviour.

    Streams the workflow events to collect ``ToolCall`` events (name + kwargs), then
    awaits the handler for the final output. Tool calls collected before a mid-run
    error survive (salvage); errors are recorded in ``error``, never raised.
    """
    result_out = AgentRunResult()
    calls: list[ToolCallRecord] = []
    try:
        # agent.run() itself can raise synchronously (pydantic/workflow validation),
        # so build the handler INSIDE the try — errors are recorded, never raised.
        handler = agent.run(user_msg=user_input)
        async for ev in handler.stream_events():
            # A ToolCall event marks a tool invocation; ToolCallResult (a different
            # class) carries the result — match ToolCall exactly so results aren't
            # double-counted.
            if type(ev).__name__ == "ToolCall":
                name = getattr(ev, "tool_name", None)
                if isinstance(name, str) and name:
                    kwargs = getattr(ev, "tool_kwargs", None)
                    # Serialize as JSON (the tool_calls QuerySpec advertises JSON) —
                    # tool_kwargs is a dict; a Python-repr string wouldn't json.loads.
                    arguments = json.dumps(kwargs if isinstance(kwargs, dict) else {}, default=str)
                    calls.append(ToolCallRecord(name=name, arguments=arguments))
        output = await handler
        result_out.final_response = str(output) if output is not None else ""
    except Exception as exc:  # noqa: BLE001 - recorded; tool calls salvaged from the stream
        result_out.error = f"{type(exc).__name__}: {exc}"

    result_out.tool_calls = calls
    return result_out


__all__ = ["ToolCallRecord", "AgentRunResult", "run_agent_capture"]
