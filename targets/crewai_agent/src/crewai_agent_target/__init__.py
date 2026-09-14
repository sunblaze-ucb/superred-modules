"""CrewAI agent target for superred.

Wraps a ``crewai.Crew`` so red-team claims/optimizers can drive it across the
crew's real attack surfaces — ``user_input`` (direct), ``tool_output`` (indirect
injection via tool returns), and ``system_prompt`` (attacker text in the agent's
backstory, which CrewAI renders into its system prompt) — and captures the final
output and the tools the agent called. Offline-testable via a scripted ``BaseLLM``.

The ``demo`` module (the scripted LLM and reference crew) imports ``crewai`` at
module load, so its symbols are re-exported *lazily* via module ``__getattr__``:
importing ``crewai_agent_target`` / ``.target`` / ``.injection`` needs no ``crewai``
install, while ``from crewai_agent_target import build_demo_crew`` pulls it in on
demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from superred.core.controller import TargetFactory

from crewai_agent_target.injection import InjectionSpec
from crewai_agent_target.runner import CrewRunResult, ToolCallRecord, run_crew_capture
from crewai_agent_target.target import (
    AGENT_RESPONSE_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_OUTPUT_TAG,
    USER_INPUT_TAG,
    CrewAIAgentTarget,
    CrewFactory,
)

if TYPE_CHECKING:
    from crewai_agent_target.demo import (
        BENIGN_TOOL,
        SENSITIVE_TOOL,
        ScriptedReactLLM,
        build_demo_crew,
        final_answer,
        scripted_llm,
        tool_action,
    )

# Names re-exported from the crewai-importing demo module, resolved lazily below.
_LAZY_DEMO = frozenset(
    {
        "BENIGN_TOOL",
        "SENSITIVE_TOOL",
        "ScriptedReactLLM",
        "build_demo_crew",
        "final_answer",
        "scripted_llm",
        "tool_action",
    }
)


def crewai_agent_target_factory(
    *,
    crew_factory: CrewFactory,
    llm: object,
    concurrency: int = 1,
) -> TargetFactory:
    """Build a :class:`TargetFactory` of fresh :class:`CrewAIAgentTarget` instances.

    ``crew_factory`` is a callable ``(llm, injection_spec) -> crewai.Crew`` (see
    :class:`CrewAIAgentTarget`); ``llm`` is required (a configured ``crewai.LLM`` or
    an offline ``BaseLLM``).
    """
    return TargetFactory(
        create=lambda: CrewAIAgentTarget(crew_factory=crew_factory, llm=llm),
        concurrency=concurrency,
    )


def __getattr__(name: str) -> Any:
    # PEP 562: resolve demo re-exports on first access so importing this package
    # (and thus crewai_agent_target.target / .injection) needs no crewai install.
    if name in _LAZY_DEMO:
        from crewai_agent_target import demo

        return getattr(demo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CrewAIAgentTarget",
    "CrewFactory",
    "InjectionSpec",
    "crewai_agent_target_factory",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    "TOOL_OUTPUT_TAG",
    "SYSTEM_PROMPT_TAG",
    "AGENT_RESPONSE_TAG",
    # runner
    "CrewRunResult",
    "ToolCallRecord",
    "run_crew_capture",
    # demo / scripted llm (lazy — require crewai)
    "BENIGN_TOOL",
    "SENSITIVE_TOOL",
    "ScriptedReactLLM",
    "build_demo_crew",
    "tool_action",
    "final_answer",
    "scripted_llm",
]
