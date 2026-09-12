"""CrewAI agent target for superred.

Wraps a ``crewai.Crew`` so red-team claims/optimizers can drive it via the
attacker-controlled ``user_input`` (injected into the task); captures the final
output and the tools the agent called. Offline-testable via a scripted ``BaseLLM``.
"""

from __future__ import annotations

from superred.core.controller import TargetFactory

from crewai_agent_target.demo import (
    BENIGN_TOOL,
    SENSITIVE_TOOL,
    ScriptedReactLLM,
    build_demo_crew,
    final_answer,
    scripted_llm,
    tool_action,
)
from crewai_agent_target.runner import CrewRunResult, ToolCallRecord, run_crew_capture
from crewai_agent_target.target import (
    SYSTEM_TAG,
    USER_INPUT_TAG,
    CrewAIAgentTarget,
    CrewFactory,
)


def crewai_agent_target_factory(
    *,
    crew_factory: CrewFactory,
    llm: object,
    concurrency: int = 1,
) -> TargetFactory:
    """Build a :class:`TargetFactory` of fresh :class:`CrewAIAgentTarget` instances.

    ``llm`` is required (a configured ``crewai.LLM`` or an offline ``BaseLLM``).
    """
    return TargetFactory(
        create=lambda: CrewAIAgentTarget(crew_factory=crew_factory, llm=llm),
        concurrency=concurrency,
    )


__all__ = [
    "CrewAIAgentTarget",
    "CrewFactory",
    "crewai_agent_target_factory",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    # runner
    "CrewRunResult",
    "ToolCallRecord",
    "run_crew_capture",
    # demo / scripted llm
    "BENIGN_TOOL",
    "SENSITIVE_TOOL",
    "ScriptedReactLLM",
    "build_demo_crew",
    "tool_action",
    "final_answer",
    "scripted_llm",
]
