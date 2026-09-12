"""mcp_tool_injection_claim: red-team the MCP tool supply chain.

One Task per tool-poisoning payload against the ``mcp_agent`` target. A poisoned
tool description carries hidden instructions; the claim scores whether the agent
follows them and calls a sensitive tool it should not (higher = more-successful
attack). Pairs with the ``goal_passthrough`` optimizer (inject the poison
verbatim) or a jailbreak optimizer (craft an evasive poison). Fully offline
against the bundled in-memory demo server + a mock LLM via
:func:`mcp_tool_injection_demo`.
"""

from __future__ import annotations

from mcp_tool_injection_claim.factory import mcp_tool_injection_claim, mcp_tool_injection_demo
from mcp_tool_injection_claim.poisons import DEFAULT_POISONS, Poison, default_poisons
from mcp_tool_injection_claim.task import (
    MCPToolInjectionNotScorableError,
    MCPToolInjectionTask,
)

__all__ = [
    "Poison",
    "DEFAULT_POISONS",
    "default_poisons",
    "MCPToolInjectionTask",
    "MCPToolInjectionNotScorableError",
    "mcp_tool_injection_claim",
    "mcp_tool_injection_demo",
]
