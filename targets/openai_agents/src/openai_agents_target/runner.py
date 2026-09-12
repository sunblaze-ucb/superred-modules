"""Run an OpenAI Agents SDK ``Agent`` and capture what it did.

Kept separate from the Target so it is unit-testable with a scripted model and a
real ``Agent`` — no network. Captures the final output, the tools the agent
called, and whether a guardrail tripwire fired (a tripwire raises out of
``Runner.run``; catching it is how we record a blocked attack).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunConfig,
    Runner,
)


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
    guardrail_tripped: bool = False
    guardrail_stage: str = ""  # "input" / "output" / "" — which guardrail fired
    error: str = ""

    @property
    def called_tool_names(self) -> list[str]:
        return [c.name for c in self.tool_calls]


def _extract_tool_calls(new_items: list[Any]) -> list[ToolCallRecord]:
    calls: list[ToolCallRecord] = []
    for item in new_items:
        if getattr(item, "type", None) == "tool_call_item":
            raw = getattr(item, "raw_item", None)
            name = getattr(raw, "name", None)
            if isinstance(name, str):
                calls.append(
                    ToolCallRecord(name=name, arguments=str(getattr(raw, "arguments", "")))
                )
    return calls


async def run_agent_capture(
    *,
    agent: Agent[Any],
    user_input: str,
    model: Any | None = None,
    max_turns: int = 10,
) -> AgentRunResult:
    """Run ``agent`` on ``user_input`` and capture its behaviour.

    ``model`` (a string id or a ``Model`` instance) overrides the agent's model
    via ``RunConfig`` — pass a scripted model for offline tests, a real model id
    (or a litellm model) for live runs. Guardrail tripwires and any run error are
    recorded rather than raised.
    """
    run_config = RunConfig(model=model, tracing_disabled=True)
    result_out = AgentRunResult()
    try:
        result = await Runner.run(agent, user_input, run_config=run_config, max_turns=max_turns)
    except InputGuardrailTripwireTriggered:
        result_out.guardrail_tripped = True
        result_out.guardrail_stage = "input"
        return result_out
    except OutputGuardrailTripwireTriggered:
        result_out.guardrail_tripped = True
        result_out.guardrail_stage = "output"
        return result_out
    except Exception as exc:  # noqa: BLE001 - recorded as error, never raised
        result_out.error = f"{type(exc).__name__}: {exc}"
        return result_out

    result_out.tool_calls = _extract_tool_calls(list(result.new_items))
    final = result.final_output
    result_out.final_response = str(final) if final is not None else ""
    return result_out


__all__ = ["ToolCallRecord", "AgentRunResult", "run_agent_capture"]
