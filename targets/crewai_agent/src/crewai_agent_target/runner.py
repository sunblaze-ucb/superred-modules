"""Run a CrewAI ``Crew`` and capture what it did.

Kept separate from the Target so it is unit-testable with a scripted ``BaseLLM``
and a real ``crewai.Crew`` — no network. Tool calls are captured via the crew's
``step_callback`` (which fires on each agent action DURING ``kickoff``), so calls
made before a mid-run error are salvaged automatically. Any error is recorded
rather than raised, so a claim can abstain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    """One tool call the crew's agent made."""

    name: str
    arguments: str


@dataclass
class CrewRunResult:
    """The captured outcome of one crew run."""

    final_response: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: str = ""

    @property
    def called_tool_names(self) -> list[str]:
        return [c.name for c in self.tool_calls]


async def run_crew_capture(*, crew: Any, user_input: str) -> CrewRunResult:
    """Run ``crew`` with ``user_input`` (filling the task's ``{user_input}``) and
    capture the tools the agent called and the final output."""
    result_out = CrewRunResult()
    calls: list[ToolCallRecord] = []

    def _recorder(step: Any) -> None:
        # CrewAI emits an AgentAction (with .tool / .tool_input) per tool use, and a
        # ToolResult (no .tool) per result; record the former.
        tool = getattr(step, "tool", None)
        if isinstance(tool, str) and tool:
            calls.append(
                ToolCallRecord(name=tool, arguments=str(getattr(step, "tool_input", "") or ""))
            )

    # Install the recorder at the AGENT level. CrewAI only copies a crew-level
    # step_callback to agents that DON'T already have one, so a custom crew_factory
    # that gives its agent a step_callback would silently drop our tool capture if we
    # only set it on the crew. Chain onto any existing agent callback so both fire.
    agents = getattr(crew, "agents", None) or []
    if agents:
        for agent in agents:
            existing = getattr(agent, "step_callback", None)
            if callable(existing) and existing is not _recorder:

                def _chained(step: Any, _orig: Any = existing) -> None:
                    _orig(step)
                    _recorder(step)

                agent.step_callback = _chained
            else:
                agent.step_callback = _recorder
    else:
        try:
            crew.step_callback = _recorder
        except Exception:  # noqa: BLE001 - not settable: tool capture degrades
            pass

    try:
        output = await crew.kickoff_async(inputs={"user_input": user_input})
        result_out.final_response = str(output) if output is not None else ""
    except Exception as exc:  # noqa: BLE001 - recorded; tool calls salvaged from the callback
        result_out.error = f"{type(exc).__name__}: {exc}"

    # calls captured by the callback during kickoff survive even if kickoff raised.
    result_out.tool_calls = calls
    return result_out


__all__ = ["ToolCallRecord", "CrewRunResult", "run_crew_capture"]
