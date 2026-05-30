"""AgentHarm tool resolver: resolves real tools, rejects unknown names."""

from __future__ import annotations

import pytest
from inspect_ai.tool import Tool

from security_claim_agentharm.agentharm_tools import (
    ALL_HARMFUL_TOOL_NAMES,
    agentharm_tool_resolver,
)


def test_resolver_returns_tool() -> None:
    tool = agentharm_tool_resolver("send_email")
    assert isinstance(tool, Tool)


def test_resolver_unknown_raises() -> None:
    with pytest.raises(ValueError, match="not available"):
        agentharm_tool_resolver("definitely_not_a_real_tool")


def test_all_tool_names_nonempty_and_includes_send_email() -> None:
    assert "send_email" in ALL_HARMFUL_TOOL_NAMES
    assert len(ALL_HARMFUL_TOOL_NAMES) > 50
