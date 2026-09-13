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
    ToolInputGuardrailTripwireTriggered,
    ToolOutputGuardrailTripwireTriggered,
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
    # Recognizes function-tool calls (``raw_item.name``), which is what agents
    # built from ``function_tool`` produce. Hosted tools (web/file search,
    # computer-use, hosted MCP) use different ``raw_item`` shapes and are not
    # captured here — point the target at function-tool agents for tool-call scoring.
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
    except InputGuardrailTripwireTriggered as exc:
        result_out.guardrail_tripped = True
        result_out.guardrail_stage = "input"
        _salvage_tool_calls(result_out, exc)
        return result_out
    except OutputGuardrailTripwireTriggered as exc:
        # Output guardrails run AFTER the agent loop, so tool calls (incl. a
        # sensitive one) may already have happened before the block — salvage them
        # so a blocked output isn't mis-scored as a clean defense.
        result_out.guardrail_tripped = True
        result_out.guardrail_stage = "output"
        _salvage_tool_calls(result_out, exc)
        return result_out
    except ToolInputGuardrailTripwireTriggered as exc:
        # Tool-level guardrails (function_tool(tool_input_guardrails=...)) are a
        # sibling AgentsException, not a subclass of the two above; a tripwire here
        # is still a DEFENDED outcome, not an error/abstain.
        result_out.guardrail_tripped = True
        result_out.guardrail_stage = "tool_input"
        _salvage_tool_calls(result_out, exc)
        return result_out
    except ToolOutputGuardrailTripwireTriggered as exc:
        result_out.guardrail_tripped = True
        result_out.guardrail_stage = "tool_output"
        _salvage_tool_calls(result_out, exc)
        return result_out
    except Exception as exc:  # noqa: BLE001 - recorded as error, never raised
        result_out.error = f"{type(exc).__name__}: {exc}"
        _salvage_tool_calls(result_out, exc)  # e.g. MaxTurnsExceeded after a tool call
        return result_out

    result_out.tool_calls = _extract_tool_calls(list(result.new_items))
    final = result.final_output
    result_out.final_response = str(final) if final is not None else ""
    return result_out


def _salvage_tool_calls(result_out: AgentRunResult, exc: BaseException) -> None:
    """Recover any tool calls the SDK attached to a partial run on an exception.

    The Agents SDK attaches the partial run (with ``new_items``) to
    ``AgentsException.run_data``, so a guardrail block / late error that fired
    after the agent already called tools still exposes those calls.
    """
    run_data = getattr(exc, "run_data", None)
    new_items = getattr(run_data, "new_items", None)
    if new_items is not None:
        result_out.tool_calls = _extract_tool_calls(list(new_items))


__all__ = ["ToolCallRecord", "AgentRunResult", "run_agent_capture"]
