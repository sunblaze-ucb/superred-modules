"""The MCP-tool-using agent loop and the MCP<->litellm tool mapping.

Kept separate from the Target so the loop is unit-testable in isolation with a
mocked completion function and a stub MCP session.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class ToolInfo(Protocol):
    """The subset of ``mcp.types.Tool`` this module reads."""

    name: str
    description: str | None
    input_schema: dict[str, Any]


class ToolResult(Protocol):
    """The subset of ``mcp.types.CallToolResult`` this module reads."""

    content: list[Any]
    is_error: bool


class MCPSession(Protocol):
    """The subset of ``mcp.ClientSession`` this module uses."""

    async def list_tools(self) -> Any: ...
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult: ...


# A completion function with litellm's shape: (model, messages, tools, **kw) -> response.
CompletionFn = Callable[..., Awaitable[Any]]


@dataclass
class ToolCallRecord:
    """One tool call the agent made during a run."""

    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool


@dataclass
class AgentRun:
    """The captured outcome of one agent run."""

    final_response: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    error: str = ""

    @property
    def called_tool_names(self) -> list[str]:
        return [c.name for c in self.tool_calls]


def result_text(result: ToolResult) -> str:
    """Concatenate the text parts of a CallToolResult (ignoring non-text content)."""
    parts: list[str] = []
    for item in result.content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def poison_tool_description(base: str | None, injection: str) -> str:
    """Append an attacker-controlled instruction to a tool description.

    This models a compromised/malicious MCP server: the tool's advertised
    description carries hidden instructions the agent may follow (tool poisoning /
    tool-supply-chain injection).
    """
    base = base or ""
    if not injection:
        return base
    sep = "\n\n" if base else ""
    return f"{base}{sep}{injection}"


def mcp_tools_to_litellm(
    tools: list[ToolInfo],
    *,
    poison_tool: str = "",
    poison_injection: str = "",
) -> list[dict[str, Any]]:
    """Map MCP tools to litellm function-calling specs, optionally poisoning one.

    If ``poison_tool`` names a tool and ``poison_injection`` is non-empty, that
    tool's advertised description is augmented with the injection.
    """
    specs: list[dict[str, Any]] = []
    for tool in tools:
        description = tool.description or ""
        if poison_injection and tool.name == poison_tool:
            description = poison_tool_description(description, poison_injection)
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": description,
                    "parameters": tool.input_schema or {"type": "object", "properties": {}},
                },
            }
        )
    return specs


async def run_agent(
    *,
    session: MCPSession,
    complete: CompletionFn,
    model: str,
    system_prompt: str,
    user_message: str,
    poison_tool: str = "",
    poison_injection: str = "",
    max_turns: int = 4,
    completion_kwargs: dict[str, Any] | None = None,
) -> AgentRun:
    """Run a bounded MCP-tool-using agent loop and capture what it did.

    Lists the server's tools (poisoning one if asked), then loops: ask the model
    with the tools available; execute any tool calls it requests via the MCP
    session; feed the results back; stop when the model answers without a tool
    call or ``max_turns`` is reached.
    """
    run = AgentRun()
    completion_kwargs = completion_kwargs or {}

    try:
        listed = await session.list_tools()
        tools: list[ToolInfo] = list(listed.tools)
    except Exception as exc:  # noqa: BLE001 - surfaced as run.error, never raised
        run.error = f"list_tools failed: {type(exc).__name__}: {exc}"
        return run

    tool_specs = mcp_tools_to_litellm(
        tools, poison_tool=poison_tool, poison_injection=poison_injection
    )

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    for _ in range(max(1, max_turns)):
        run.turns += 1
        try:
            response = await complete(
                model=model, messages=messages, tools=tool_specs, **completion_kwargs
            )
            message = response.choices[0].message
        except Exception as exc:  # noqa: BLE001
            run.error = f"completion failed: {type(exc).__name__}: {exc}"
            break

        tool_calls = getattr(message, "tool_calls", None) or []
        content = getattr(message, "content", None) or ""

        # record the assistant turn on the wire
        assistant_entry: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": getattr(tc, "id", ""),
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)
        run.transcript.append(assistant_entry)

        if not tool_calls:
            run.final_response = content
            break

        # execute each requested tool call against the MCP server
        for tc in tool_calls:
            name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            try:
                result = await session.call_tool(name, arguments)
                text = result_text(result)
                is_error = bool(result.is_error)
            except Exception as exc:  # noqa: BLE001
                text = f"tool call error: {type(exc).__name__}: {exc}"
                is_error = True
            run.tool_calls.append(
                ToolCallRecord(name=name, arguments=arguments, result=text, is_error=is_error)
            )
            tool_entry = {
                "role": "tool",
                "tool_call_id": getattr(tc, "id", ""),
                "name": name,
                "content": text,
            }
            messages.append(tool_entry)
            run.transcript.append(tool_entry)

    return run


__all__ = [
    "ToolCallRecord",
    "AgentRun",
    "result_text",
    "poison_tool_description",
    "mcp_tools_to_litellm",
    "run_agent",
]
