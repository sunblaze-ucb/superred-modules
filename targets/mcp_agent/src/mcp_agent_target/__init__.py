"""MCP agent target for superred: an LLM agent that uses MCP tools, red-teamed
for tool-supply-chain (tool-poisoning) attacks.

The attacker controls a poison instruction injected into a tool's advertised
description (``tool_poison``) and the direct ``user_message``; the paired
``mcp_tool_injection`` claim scores whether the agent then calls a sensitive tool
it should not. Uniquely e2e-verifiable in CI via an in-memory MCP server.
"""

from __future__ import annotations

from superred.core.controller import TargetFactory

from mcp_agent_target.agent import (
    AgentRun,
    ToolCallRecord,
    mcp_tools_to_litellm,
    poison_tool_description,
    run_agent,
)
from mcp_agent_target.servers import (
    BENIGN_TOOL,
    SENSITIVE_TOOL,
    build_demo_server,
    http_session_provider,
    in_memory_session_provider,
    stdio_session_provider,
)
from mcp_agent_target.target import (
    SYSTEM_TAG,
    TOOL_SUPPLY_CHAIN_TAG,
    USER_INPUT_TAG,
    MCPAgentTarget,
    SessionProvider,
)


def mcp_agent_target_factory(
    *,
    model: str,
    session_provider: SessionProvider,
    api_base: str = "",
    api_key: str = "",
    max_turns: int = 4,
    concurrency: int = 1,
) -> TargetFactory:
    """Build a :class:`TargetFactory` of fresh :class:`MCPAgentTarget` instances."""
    return TargetFactory(
        create=lambda: MCPAgentTarget(
            model=model,
            session_provider=session_provider,
            api_base=api_base,
            api_key=api_key,
            max_turns=max_turns,
        ),
        concurrency=concurrency,
    )


__all__ = [
    "MCPAgentTarget",
    "SessionProvider",
    "mcp_agent_target_factory",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    "TOOL_SUPPLY_CHAIN_TAG",
    # agent loop
    "AgentRun",
    "ToolCallRecord",
    "run_agent",
    "mcp_tools_to_litellm",
    "poison_tool_description",
    # servers
    "BENIGN_TOOL",
    "SENSITIVE_TOOL",
    "build_demo_server",
    "in_memory_session_provider",
    "stdio_session_provider",
    "http_session_provider",
]
