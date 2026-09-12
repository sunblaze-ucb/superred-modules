"""OpenAI Agents SDK agent target for superred.

Wraps an ``agents.Agent`` so red-team claims/optimizers can drive it via the
attacker-controlled ``user_input``; captures the final output, tool calls, and
guardrail tripwires. Offline-testable via a scripted ``Model``.
"""

from __future__ import annotations

from superred.core.controller import TargetFactory

from openai_agents_target.demo import (
    BENIGN_TOOL,
    SENSITIVE_TOOL,
    ScriptedModel,
    build_demo_agent,
    function_call_output,
    message_output,
)
from openai_agents_target.runner import AgentRunResult, ToolCallRecord, run_agent_capture
from openai_agents_target.target import (
    SYSTEM_TAG,
    USER_INPUT_TAG,
    AgentFactory,
    OpenAIAgentTarget,
)


def openai_agent_target_factory(
    *,
    agent_factory: AgentFactory,
    model: object | None = None,
    max_turns: int = 10,
    concurrency: int = 1,
) -> TargetFactory:
    """Build a :class:`TargetFactory` of fresh :class:`OpenAIAgentTarget` instances."""
    return TargetFactory(
        create=lambda: OpenAIAgentTarget(
            agent_factory=agent_factory, model=model, max_turns=max_turns
        ),
        concurrency=concurrency,
    )


__all__ = [
    "OpenAIAgentTarget",
    "AgentFactory",
    "openai_agent_target_factory",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    # runner
    "AgentRunResult",
    "ToolCallRecord",
    "run_agent_capture",
    # demo / scripted model
    "BENIGN_TOOL",
    "SENSITIVE_TOOL",
    "ScriptedModel",
    "build_demo_agent",
    "message_output",
    "function_call_output",
]
