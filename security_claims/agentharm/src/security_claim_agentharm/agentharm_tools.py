"""AgentHarm tool resolver: the benchmark-specific seam for the general target.

The general :class:`InspectAgentTarget` is constructed with a
``tool_resolver(name) -> Tool``.  For AgentHarm that resolver maps a tool name
to the upstream harmful tool factory (``inspect_evals.agentharm.benchmark
.harmful_tools``), reproducing exactly what upstream's
``setup_tools_from_metadata`` does (``getattr(tools_module, name)()``).
"""

from __future__ import annotations

from typing import cast

from inspect_ai.tool import Tool
from inspect_evals.agentharm.benchmark import harmful_tools
from inspect_evals.agentharm.utils import get_benchmark_tools

ALL_HARMFUL_TOOL_NAMES: frozenset[str] = frozenset(get_benchmark_tools("harmful"))
"""Every harmful tool name shipped by upstream AgentHarm."""


def agentharm_tool_resolver(name: str) -> Tool:
    """Resolve an AgentHarm harmful tool name to an inspect ``Tool``.

    Mirrors upstream ``getattr(harmful_tools, name)()``.

    Raises:
        ValueError: if the tool name is not an AgentHarm harmful tool.
    """
    factory = getattr(harmful_tools, name, None)
    if factory is None:
        raise ValueError(f"AgentHarm harmful tool {name!r} is not available.")
    return cast(Tool, factory())


__all__ = ["agentharm_tool_resolver", "ALL_HARMFUL_TOOL_NAMES"]
