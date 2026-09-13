"""Run an AG2 (AutoGen) ``Agent`` and capture what it did.

Kept separate from the Target so it is unit-testable with a scripted model
config and a real ``ag2.Agent`` — no network. Captures the final output text and
the tools the agent called (read from the run's event history). Tool calls made
before a mid-run error are salvaged from the run stream, so a looping / partially
failed sensitive-tool misuse is not lost. Any error is recorded, never raised.
"""

from __future__ import annotations

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


def _extract_tool_calls(events: list[Any]) -> list[ToolCallRecord]:
    # Lazy framework import: keeps the module importable without ag2 installed
    # (only real runs, which need ag2 anyway, reach here). ``ag2.testing``
    # re-exports the identical class (``ag2.testing.ToolCallEvent is
    # ag2.events.tool_events.ToolCallEvent``), so this isinstance check matches
    # events produced by scripted (demo) and real runs alike.
    from ag2.events.tool_events import ToolCallEvent

    calls: list[ToolCallRecord] = []
    for e in events:
        if isinstance(e, ToolCallEvent):
            name = getattr(e, "name", None)
            if isinstance(name, str):
                calls.append(
                    ToolCallRecord(name=name, arguments=str(getattr(e, "arguments", "") or ""))
                )
    return calls


async def run_agent_capture(*, agent: Any, user_input: str) -> AgentRunResult:
    """Run ``agent`` on ``user_input`` and capture its behaviour.

    AG2's ``AgentRun`` must be driven inside its ``async with`` block; the final
    reply and the event history are read there. If the run raises mid-way (a tool
    or model error), the events already emitted are salvaged from the run stream so
    the tool calls the agent made are not lost. Errors are recorded in ``error``.
    """
    result_out = AgentRunResult()
    events: list[Any] = []
    content: Any = ""
    try:
        async with agent.run(user_input) as run:
            try:
                reply = await run.result()
                content = await reply.content()
                events = list(await reply.history.get_events())
            except Exception as exc:  # noqa: BLE001 - recorded; partial events salvaged
                result_out.error = f"{type(exc).__name__}: {exc}"
                try:
                    events = list(await run.stream.history.get_events())
                except Exception:  # noqa: BLE001 - nothing to salvage
                    events = []
    except Exception as exc:  # noqa: BLE001 - e.g. the run's __aexit__ raising on teardown
        if not result_out.error:
            result_out.error = f"{type(exc).__name__}: {exc}"
    finally:
        # Populate from whatever was captured, on EVERY exit path. ``events`` and
        # ``content`` are materialized inside the ``async with`` body (before the
        # run's __aexit__ runs), so salvaged tool calls survive even if __aexit__
        # raises during teardown of an already-errored run — never dropped.
        result_out.tool_calls = _extract_tool_calls(events)
        if isinstance(content, str):
            result_out.final_response = content
        elif content is not None:
            result_out.final_response = str(content)
    return result_out


__all__ = ["ToolCallRecord", "AgentRunResult", "run_agent_capture"]
