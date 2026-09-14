"""Run a LangChain (v1) ``create_agent`` graph and capture what it did.

Kept separate from the Target so it is unit-testable with a scripted model and a
real agent graph — no network. Captures the final output text and the tools the
agent called (read back from the returned message list); any run error is
recorded rather than raised, so a claim can abstain instead of mis-scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage


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


def _extract(messages: list[Any]) -> tuple[list[ToolCallRecord], str]:
    """Pull tool calls and the final assistant text out of the returned messages.

    Tool calls come from ``AIMessage.tool_calls`` (LangChain's normalized shape:
    dicts with ``name``/``args``), collected across every turn. The final response
    is the ``.text`` of the *last* message when it is an ``AIMessage`` — ``.text``
    flattens list/content-block messages (Claude, multimodal, reasoning) to plain
    text, and taking only the last turn avoids reporting a mid-run utterance as the
    answer. A run that ends on a tool turn (recursion limit, no final synthesis)
    has no final text, so ``final_response`` is empty.
    """
    calls: list[ToolCallRecord] = []
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for tc in m.tool_calls or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            if isinstance(name, str):
                calls.append(
                    ToolCallRecord(name=name, arguments=json.dumps(args or {}, default=str))
                )
    final = ""
    if messages and isinstance(messages[-1], AIMessage):
        text = messages[-1].text
        final = text if isinstance(text, str) else str(text)
    return calls, final


async def run_agent_capture(
    *,
    agent: Any,
    user_input: str,
    recursion_limit: int = 25,
) -> AgentRunResult:
    """Run ``agent`` (a compiled ``create_agent`` graph) on ``user_input`` and
    capture its behaviour.

    Streams state values (``astream(stream_mode="values")``) rather than a single
    ``ainvoke`` so that if the run raises mid-way — most importantly a
    ``GraphRecursionError`` from a tool-call loop — the tool calls the agent
    already made are salvaged from the last state, not lost. A looping
    sensitive-tool misuse must not read as an empty ("refused") run. Any error is
    recorded in ``error`` rather than raised; the partial state is still extracted.
    """
    result_out = AgentRunResult()
    last_state: Any = None
    try:
        async for state in agent.astream(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            last_state = state
    except Exception as exc:  # noqa: BLE001 - recorded; partial state salvaged below
        result_out.error = f"{type(exc).__name__}: {exc}"

    messages = last_state.get("messages", []) if isinstance(last_state, dict) else []
    result_out.tool_calls, result_out.final_response = _extract(list(messages))
    return result_out


__all__ = ["ToolCallRecord", "AgentRunResult", "run_agent_capture"]
