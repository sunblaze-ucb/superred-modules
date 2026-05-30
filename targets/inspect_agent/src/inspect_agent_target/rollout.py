"""The generic agent rollout: one tool-calling loop over a per-turn tool set.

This reproduces the body of inspect's ``generate(tool_calls="loop")`` using
inspect's own primitives (``Model.generate`` + ``execute_tools``), so the
resulting ``list[ChatMessage]`` is structurally what an ``inspect_ai.eval``
run would produce.  Keeping the loop here (rather than driving a full
``inspect_ai.eval``) lets the superred Target own orchestration while the
message trace stays faithful.

Tools are supplied by a ``tools_provider`` called once per turn, BEFORE each
model call.  This is the seam that lets the target fire its tool-catalogue
Controllables every turn (so an attacker-scoped optimizer can edit the
catalogue mid-run) while keeping this loop benchmark-agnostic.  Callers that do
not edit tools can use :func:`static_tools_provider`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    Model,
    execute_tools,
)
from inspect_ai.tool import Tool

# inspect's standard tool-choice strings.
ToolChoice = Literal["auto", "any", "none"]

ToolsProvider = Callable[[], Awaitable[Sequence[Tool]]]


def static_tools_provider(tools: Sequence[Tool]) -> ToolsProvider:
    """A tools provider that returns a fixed tool list every turn."""
    fixed = list(tools)

    async def provider() -> Sequence[Tool]:
        return fixed

    return provider


async def run_rollout(
    model: Model,
    *,
    system_prompt: str,
    user_prompt: str,
    tools_provider: ToolsProvider,
    tool_choice: ToolChoice = "auto",
    message_limit: int,
) -> list[ChatMessage]:
    """Run the tool-calling loop and return the full message list.

    Args:
        model: an inspect ``Model`` (already bound to its generation config).
        system_prompt: system message text; empty string means none.
        user_prompt: the user instruction the agent acts on.
        tools_provider: async callable returning the tools for the current turn.
            Called once per turn before the model call; the same tools are used
            for that turn's generation and tool execution.
        tool_choice: one of "auto", "any", "none".
        message_limit: cap on total messages (system + user + assistant + tool).

    Returns:
        The full ``list[ChatMessage]`` produced by the rollout.
    """
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessageSystem(content=system_prompt))
    messages.append(ChatMessageUser(content=user_prompt))

    while True:
        tools = list(await tools_provider())
        output = await model.generate(messages, tools=tools, tool_choice=tool_choice)
        messages.append(output.message)
        if not output.message.tool_calls:
            break
        # execute_tools runs the last assistant message's tool calls against the
        # SAME tools the model saw this turn and returns the tool-result messages.
        result = await execute_tools(messages, tools)
        messages.extend(result.messages)
        if len(messages) >= message_limit:
            break

    return messages


__all__ = ["run_rollout", "static_tools_provider", "ToolChoice", "ToolsProvider"]
