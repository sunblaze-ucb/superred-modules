"""LangChain (v1) agent target for superred.

Wraps a LangChain ``create_agent`` graph so red-team claims/optimizers can drive
it via the attacker-controlled ``user_input``; captures the final output and the
tools the agent called. Offline-testable via a scripted ``BaseChatModel``.
"""

from __future__ import annotations

from superred.core.controller import TargetFactory

from langchain_agent_target.demo import (
    BENIGN_TOOL,
    SENSITIVE_TOOL,
    ScriptedChatModel,
    build_demo_agent,
    message_output,
    tool_call_output,
)
from langchain_agent_target.runner import AgentRunResult, ToolCallRecord, run_agent_capture
from langchain_agent_target.target import (
    SYSTEM_TAG,
    USER_INPUT_TAG,
    AgentFactory,
    LangChainAgentTarget,
)


def langchain_agent_target_factory(
    *,
    agent_factory: AgentFactory,
    model: object,
    recursion_limit: int = 25,
    concurrency: int = 1,
) -> TargetFactory:
    """Build a :class:`TargetFactory` of fresh :class:`LangChainAgentTarget` instances.

    ``model`` is required (a model id string or a ``BaseChatModel``); LangChain's
    ``create_agent`` has no default model.
    """
    return TargetFactory(
        create=lambda: LangChainAgentTarget(
            agent_factory=agent_factory, model=model, recursion_limit=recursion_limit
        ),
        concurrency=concurrency,
    )


__all__ = [
    "LangChainAgentTarget",
    "AgentFactory",
    "langchain_agent_target_factory",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    # runner
    "AgentRunResult",
    "ToolCallRecord",
    "run_agent_capture",
    # demo / scripted model
    "BENIGN_TOOL",
    "SENSITIVE_TOOL",
    "ScriptedChatModel",
    "build_demo_agent",
    "message_output",
    "tool_call_output",
]
