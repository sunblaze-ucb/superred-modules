"""LlamaIndex agent target for superred.

Wraps a LlamaIndex ``ReActAgent`` so red-team claims/optimizers can drive it across
the agent's real attack surfaces — ``user_input`` (direct), ``tool_output``
(indirect injection via tool returns), and ``system_prompt`` — and captures the
final output and the tools the agent called. Offline-testable via a scripted
``CustomLLM``.

The ``target`` / ``injection`` / ``runner`` modules import ``llama-index`` lazily,
so they (and this package) import without the framework installed. The ``demo``
module — a scripted ``CustomLLM`` subclass and the reference agent factory — needs
the framework at import time, so its symbols are loaded on first access (PEP 562)
rather than at package import, keeping the framework-free import path clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from superred.core.controller import TargetFactory

from llamaindex_agent_target.injection import InjectionSpec
from llamaindex_agent_target.runner import AgentRunResult, ToolCallRecord, run_agent_capture
from llamaindex_agent_target.target import (
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_OUTPUT_TAG,
    USER_INPUT_TAG,
    AgentFactory,
    LlamaIndexAgentTarget,
)

# demo.py imports llama-index at module top (ScriptedReActLLM subclasses CustomLLM),
# so these are loaded lazily to keep the package importable without the framework.
_DEMO_EXPORTS = frozenset(
    {
        "BENIGN_TOOL",
        "SENSITIVE_TOOL",
        "ScriptedReActLLM",
        "build_demo_agent",
        "final_answer",
        "scripted_llm",
        "tool_action",
    }
)

if TYPE_CHECKING:  # names for type-checkers / IDEs; resolved at runtime via __getattr__
    from llamaindex_agent_target.demo import (
        BENIGN_TOOL,
        SENSITIVE_TOOL,
        ScriptedReActLLM,
        build_demo_agent,
        final_answer,
        scripted_llm,
        tool_action,
    )


def __getattr__(name: str) -> Any:
    """Lazily resolve demo symbols on first access (PEP 562)."""
    if name in _DEMO_EXPORTS:
        from llamaindex_agent_target import demo

        return getattr(demo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "InjectionSpec",
    "llamaindex_agent_target_factory",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    "TOOL_OUTPUT_TAG",
    "SYSTEM_PROMPT_TAG",
    # runner
    "AgentRunResult",
    "ToolCallRecord",
    "run_agent_capture",
    # demo / scripted llm (lazily loaded)
    "BENIGN_TOOL",
    "SENSITIVE_TOOL",
    "ScriptedReActLLM",
    "build_demo_agent",
    "tool_action",
    "final_answer",
    "scripted_llm",
]
