"""AG2 (AutoGen) agent target for superred.

Wraps an ``ag2.Agent`` so red-team claims/optimizers can drive it across the
agent's real attack surfaces — ``user_input`` (direct), ``tool_output`` (indirect
injection via tool returns), and ``system_prompt`` — and captures the final output
and the tools the agent called. Offline-testable via ``ag2.testing.TestConfig`` (a
scripted model).
"""

from __future__ import annotations

from superred.core.controller import TargetFactory

from ag2_agent_target.demo import (
    BENIGN_TOOL,
    SENSITIVE_TOOL,
    build_demo_agent,
    message_turn,
    scripted_config,
    tool_call_turn,
)
from ag2_agent_target.injection import InjectionSpec
from ag2_agent_target.runner import AgentRunResult, ToolCallRecord, run_agent_capture
from ag2_agent_target.target import (
    AGENT_RESPONSE_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_OUTPUT_TAG,
    USER_INPUT_TAG,
    AG2AgentTarget,
    AgentFactory,
)


def ag2_agent_target_factory(
    *,
    agent_factory: AgentFactory,
    model: object,
    concurrency: int = 1,
) -> TargetFactory:
    """Build a :class:`TargetFactory` of fresh :class:`AG2AgentTarget` instances.

    ``model`` is required (an AG2 ``ModelConfig`` or an offline ``TestConfig``).
    """
    return TargetFactory(
        create=lambda: AG2AgentTarget(agent_factory=agent_factory, model=model),
        concurrency=concurrency,
    )


__all__ = [
    "AG2AgentTarget",
    "AgentFactory",
    "InjectionSpec",
    "ag2_agent_target_factory",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    "TOOL_OUTPUT_TAG",
    "SYSTEM_PROMPT_TAG",
    "AGENT_RESPONSE_TAG",
    # runner
    "AgentRunResult",
    "ToolCallRecord",
    "run_agent_capture",
    # demo / scripted config
    "BENIGN_TOOL",
    "SENSITIVE_TOOL",
    "build_demo_agent",
    "message_turn",
    "tool_call_turn",
    "scripted_config",
]
