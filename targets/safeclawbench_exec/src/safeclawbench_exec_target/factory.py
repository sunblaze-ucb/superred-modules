"""Factory for the SafeClawBench Exec-Balanced target."""

from __future__ import annotations

from superred.core.controller import TargetFactory
from superred.core.types.llm import LLMConfig

from safeclawbench_exec_target.target import AgentModel, SafeClawBenchExecTarget


def safeclawbench_exec_target_factory(
    *,
    agent_llm_config: LLMConfig | None = None,
    agent_client: AgentModel | None = None,
    defense_level: str = "D0",
    max_tool_calls: int = 4,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    concurrency: int = 1,
) -> TargetFactory:
    """Build a :class:`TargetFactory` of fresh :class:`SafeClawBenchExecTarget`.

    One of ``agent_llm_config`` (the model-under-test) or ``agent_client`` (a
    pre-built client / test stub) is required. Each instance owns its own mock
    world, so ``concurrency>1`` runs independent scenarios in parallel.
    """
    return TargetFactory(
        create=lambda: SafeClawBenchExecTarget(
            agent_llm_config=agent_llm_config,
            agent_client=agent_client,
            defense_level=defense_level,
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
        concurrency=concurrency,
    )


__all__ = ["safeclawbench_exec_target_factory"]
