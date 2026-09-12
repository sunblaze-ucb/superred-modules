"""LlamaIndex agent target for superred.

Wraps a LlamaIndex ``ReActAgent`` so red-team claims/optimizers can drive it via
the attacker-controlled ``user_input``; captures the final output and the tools
the agent called. Offline-testable via a scripted ``CustomLLM``.
"""

from __future__ import annotations

from superred.core.controller import TargetFactory

from llamaindex_agent_target.demo import (
    BENIGN_TOOL,
    SENSITIVE_TOOL,
    ScriptedReActLLM,
    build_demo_agent,
    final_answer,
    scripted_llm,
    tool_action,
)
from llamaindex_agent_target.runner import AgentRunResult, ToolCallRecord, run_agent_capture
from llamaindex_agent_target.target import (
    SYSTEM_TAG,
    USER_INPUT_TAG,
    AgentFactory,
    LlamaIndexAgentTarget,
)


def llamaindex_agent_target_factory(
    *,
    agent_factory: AgentFactory,
    llm: object,
    concurrency: int = 1,
) -> TargetFactory:
    """Build a :class:`TargetFactory` of fresh :class:`LlamaIndexAgentTarget` instances.

    ``llm`` is required (a configured LlamaIndex ``LLM`` or an offline scripted one).
    """
    return TargetFactory(
        create=lambda: LlamaIndexAgentTarget(agent_factory=agent_factory, llm=llm),
        concurrency=concurrency,
    )


__all__ = [
    "LlamaIndexAgentTarget",
    "AgentFactory",
    "llamaindex_agent_target_factory",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    # runner
    "AgentRunResult",
    "ToolCallRecord",
    "run_agent_capture",
    # demo / scripted llm
    "BENIGN_TOOL",
    "SENSITIVE_TOOL",
    "ScriptedReActLLM",
    "build_demo_agent",
    "tool_action",
    "final_answer",
    "scripted_llm",
]
