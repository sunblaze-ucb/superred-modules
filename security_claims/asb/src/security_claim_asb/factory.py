"""Factory functions building hierarchical ASB security claims.

Top factory ``asb_claim(...)`` enumerates one :class:`AsbAttackTask` per
(agent, benign task, attacker tool) triple, with filters. Convenience
roll-ups cover the natural ASB axes (per-agent, aggressive vs
non-aggressive, attack type). ``asb_target_factory(...)`` is the
convenience wiring of an :class:`AsbTarget` for these claims.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import cast

from asb_target import AsbTarget
from asb_target.memory_store import DEFAULT_EMBED_MODEL
from superred.core.controller import TargetFactory
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task
from superred.core.types.llm import LLMConfig

from security_claim_asb.categories import (
    AGGRESSIVE_ATTACK_TOOLS_PATH,
    ALL_ATTACK_TOOLS_PATH,
    ATTACK_TYPES,
    NON_AGGRESSIVE_ATTACK_TOOLS_PATH,
    load_agent_tasks,
    load_attack_tools,
)
from security_claim_asb.judge import AsbRefusalJudge, RefusalJudge
from security_claim_asb.task import AsbAttackTask


def _build_judge(
    judge: RefusalJudge | None, judge_llm_config: LLMConfig | None
) -> RefusalJudge | None:
    if judge is not None:
        return judge
    if judge_llm_config is not None:
        return AsbRefusalJudge.from_config(judge_llm_config)
    return None


def asb_claim(
    *,
    agents: Iterable[str] | None = None,
    attack_types: Iterable[str] | None = None,
    aggressive: bool | None = None,
    attacker_tools: Iterable[str] | None = None,
    tasks_per_agent: int | None = 1,
    attacker_tools_path: str = ALL_ATTACK_TOOLS_PATH,
    memory: bool = False,
    judge: RefusalJudge | None = None,
    judge_llm_config: LLMConfig | None = None,
) -> SecurityClaim[AsbTarget]:
    """Build an ASB claim.

    Args:
        agents: restrict to these scenario/agent names (default: all 10).
        attack_types: restrict to these 'Attack Type' values (Stealthy/Disruptive).
        aggressive: restrict to aggressive (True) / non-aggressive (False) tools.
        attacker_tools: restrict to these attacker-tool names (exclusive filter).
        tasks_per_agent: benign cover-tasks per agent (default 1 == ASB CLI default,
            ~400 tasks; ``None`` uses all ~5, ~2000 tasks).
        attacker_tools_path: which attacker-tool file to load.
        memory: enable the target's durable memory mode for each task (so a
            multi-run attacker can poison-then-exploit and ``memory_found`` is
            scored). Default off.
        judge / judge_llm_config: out-of-band refusal judge (gpt-4o-mini).
    """
    agent_tasks = load_agent_tasks()
    rows = load_attack_tools(attacker_tools_path)
    judge_obj = _build_judge(judge, judge_llm_config)

    agent_set = set(agents) if agents is not None else None
    type_set = set(attack_types) if attack_types is not None else None
    tool_set = set(attacker_tools) if attacker_tools is not None else None

    if agent_set is not None:
        unknown = agent_set - set(agent_tasks)
        if unknown:
            raise ValueError(f"Unknown agents: {sorted(unknown)}; known: {sorted(agent_tasks)}")
    if type_set is not None:
        unknown = type_set - set(ATTACK_TYPES)
        if unknown:
            raise ValueError(
                f"Unknown attack_types: {sorted(unknown)}; known: {list(ATTACK_TYPES)}"
            )

    by_agent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_agent[row["Corresponding Agent"]].append(row)

    tasks: list[AsbAttackTask] = []
    for agent, benign_tasks in agent_tasks.items():
        if agent_set is not None and agent not in agent_set:
            continue
        selected = benign_tasks if tasks_per_agent is None else benign_tasks[:tasks_per_agent]
        for idx, benign in enumerate(selected):
            for row in by_agent.get(agent, []):
                if tool_set is not None and row["Attacker Tool"] not in tool_set:
                    continue
                if type_set is not None and row.get("Attack Type") not in type_set:
                    continue
                if aggressive is not None:
                    is_agg = str(row.get("Aggressive", "")).strip().lower() == "true"
                    if is_agg != aggressive:
                        continue
                tasks.append(
                    AsbAttackTask(
                        agent_name=agent,
                        benign_task=benign,
                        benign_task_index=idx,
                        row=row,
                        judge=judge_obj,
                        memory=memory,
                    )
                )

    if not tasks:
        raise ValueError("asb_claim produced no tasks for the given filters")
    return SecurityClaim.from_tasks(cast(list[Task[AsbTarget]], tasks))


# -- Convenience roll-ups ----------------------------------------------------


def asb_agent_claim(agent_name: str, **kwargs: object) -> SecurityClaim[AsbTarget]:
    """All attack cases for a single scenario/agent."""
    return asb_claim(agents=[agent_name], **kwargs)  # type: ignore[arg-type]


def asb_aggressive_claim(**kwargs: object) -> SecurityClaim[AsbTarget]:
    """Only the aggressive attacker tools."""
    return asb_claim(aggressive=True, attacker_tools_path=AGGRESSIVE_ATTACK_TOOLS_PATH, **kwargs)  # type: ignore[arg-type]


def asb_non_aggressive_claim(**kwargs: object) -> SecurityClaim[AsbTarget]:
    """Only the non-aggressive attacker tools."""
    return asb_claim(
        aggressive=False,
        attacker_tools_path=NON_AGGRESSIVE_ATTACK_TOOLS_PATH,
        **kwargs,  # type: ignore[arg-type]
    )


def asb_stealthy_claim(**kwargs: object) -> SecurityClaim[AsbTarget]:
    """Only the 'Stealthy Attack' tools."""
    return asb_claim(attack_types=["Stealthy Attack"], **kwargs)  # type: ignore[arg-type]


def asb_disruptive_claim(**kwargs: object) -> SecurityClaim[AsbTarget]:
    """Only the 'Disruptive Attack' tools."""
    return asb_claim(attack_types=["Disruptive Attack"], **kwargs)  # type: ignore[arg-type]


def asb_combined_claim(claims: list[SecurityClaim[AsbTarget]]) -> SecurityClaim[AsbTarget]:
    """Compose several ASB sub-claims into one (lazy chaining)."""
    return SecurityClaim.from_claims(claims)


# -- Target wiring convenience ----------------------------------------------


def asb_target_factory(
    *,
    model: str = "gpt-4o-2024-05-13",
    api_base: str | None = None,
    api_key: str | None = None,
    request_delay_seconds: float = 2.0,
    max_output_tokens: int = 1024,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> TargetFactory:
    """A ``TargetFactory`` for the ASB target (concurrency locked to 1).

    ASB uses a process-global request queue + single scheduler, so the
    target must not be parallelized; ``concurrency=1`` is enforced. The
    memory store embeds through the same proxy (``embed_model``).
    """
    return TargetFactory(
        create=lambda: AsbTarget(
            model=model,
            api_base=api_base,
            api_key=api_key,
            request_delay_seconds=request_delay_seconds,
            max_output_tokens=max_output_tokens,
            embed_model=embed_model,
        ),
        concurrency=1,
    )


__all__ = [
    "asb_claim",
    "asb_agent_claim",
    "asb_aggressive_claim",
    "asb_non_aggressive_claim",
    "asb_stealthy_claim",
    "asb_disruptive_claim",
    "asb_combined_claim",
    "asb_target_factory",
]
