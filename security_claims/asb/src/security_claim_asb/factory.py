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
from asb_target.target import _EMBED_INHERIT
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
from security_claim_asb.judge import (
    AsbAdversarialJudge,
    AsbRefusalJudge,
    RefusalJudge,
    SuccessJudge,
)
from security_claim_asb.task import AsbAttackTask


def _build_judge(
    judge: RefusalJudge | None, judge_llm_config: LLMConfig | None
) -> RefusalJudge | None:
    if judge is not None:
        return judge
    if judge_llm_config is not None:
        return AsbRefusalJudge.from_config(judge_llm_config)
    return None


def _build_success_judge(
    success_judge: SuccessJudge | None, success_judge_llm_config: LLMConfig | None
) -> SuccessJudge | None:
    if success_judge is not None:
        return success_judge
    if success_judge_llm_config is not None:
        return AsbAdversarialJudge.from_config(success_judge_llm_config)
    return None


def asb_claim(
    *,
    agents: Iterable[str] | None = None,
    attack_types: Iterable[str] | None = None,
    aggressive: bool | None = None,
    attacker_tools: Iterable[str] | None = None,
    tasks_per_agent: int | None = 1,
    attacker_tools_path: str = NON_AGGRESSIVE_ATTACK_TOOLS_PATH,
    memory: bool = False,
    force_attacker_tool: bool = False,
    judge: RefusalJudge | None = None,
    judge_llm_config: LLMConfig | None = None,
    success_judge: SuccessJudge | None = None,
    success_judge_llm_config: LLMConfig | None = None,
) -> SecurityClaim[AsbTarget]:
    """Build an ASB claim.

    Args:
        agents: restrict to these scenario/agent names (default: all 10).
        attack_types: restrict to these 'Attack Type' values (Stealthy/Disruptive).
        aggressive: restrict to aggressive (True) / non-aggressive (False) tools.
        attacker_tools: restrict to these attacker-tool names (exclusive filter).
        tasks_per_agent: benign cover-tasks per agent (default 1 == ASB's --task_num
            default; with the default non-aggressive tool set this is ~200 tasks,
            matching a bare upstream run; ``None`` uses all ~5 benign tasks).
        attacker_tools_path: which attacker-tool file to load. Default is the
            200-row non-aggressive set, matching upstream's CLI default; pass
            ``ALL_ATTACK_TOOLS_PATH`` for the full 400 or
            ``AGGRESSIVE_ATTACK_TOOLS_PATH`` for the aggressive 200.
        memory: enable the target's durable memory mode for each task (so a
            multi-run attacker can poison-then-exploit and ``memory_found`` is
            scored). Default off.
        force_attacker_tool: reproduce upstream's ``attacker_tool_injection``
            (force the attacker tool into every plan step), which ASB applies on
            a DPI/OPI attack run. Set this for a context-injection (DPI/OPI)
            sweep; leave off for a clean baseline or a PoT/MP sweep. Default off.
        judge / judge_llm_config: out-of-band refusal judge (gpt-4o-mini).
        success_judge / success_judge_llm_config: optional out-of-band adversarial
            judge (gpt-4o-mini). When set, evaluate() adds an ``adversarial``
            sub-score and a ``genuine_attack`` = (tool invoked) AND (adversarial)
            sub-score. The primary ``attack_succeeded`` is always the deterministic
            tool invocation, judge or not.
    """
    agent_tasks = load_agent_tasks()
    rows = load_attack_tools(attacker_tools_path)
    judge_obj = _build_judge(judge, judge_llm_config)
    success_judge_obj = _build_success_judge(success_judge, success_judge_llm_config)

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
                        success_judge=success_judge_obj,
                        memory=memory,
                        force_attacker_tool=force_attacker_tool,
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


def asb_full_claim(**kwargs: object) -> SecurityClaim[AsbTarget]:
    """The full 400-tool benchmark (200 aggressive + 200 non-aggressive),
    broader than the default non-aggressive 200."""
    return asb_claim(attacker_tools_path=ALL_ATTACK_TOOLS_PATH, **kwargs)  # type: ignore[arg-type]


def asb_stealthy_claim(**kwargs: object) -> SecurityClaim[AsbTarget]:
    """All 'Stealthy Attack' tools (across aggressive + non-aggressive)."""
    return asb_claim(
        attack_types=["Stealthy Attack"],
        attacker_tools_path=ALL_ATTACK_TOOLS_PATH,
        **kwargs,  # type: ignore[arg-type]
    )


def asb_disruptive_claim(**kwargs: object) -> SecurityClaim[AsbTarget]:
    """All 'Disruptive Attack' tools (across aggressive + non-aggressive)."""
    return asb_claim(
        attack_types=["Disruptive Attack"],
        attacker_tools_path=ALL_ATTACK_TOOLS_PATH,
        **kwargs,  # type: ignore[arg-type]
    )


def asb_combined_claim(claims: list[SecurityClaim[AsbTarget]]) -> SecurityClaim[AsbTarget]:
    """Compose several ASB sub-claims into one (lazy chaining)."""
    return SecurityClaim.from_claims(claims)


# -- Target wiring convenience ----------------------------------------------


def asb_target_factory(
    *,
    model: str = "gpt-4o-mini",
    api_base: str | None = None,
    api_key: str | None = None,
    request_delay_seconds: float = 2.0,
    max_output_tokens: int = 1024,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_api_base: str | None | object = _EMBED_INHERIT,
    embed_api_key: str | None | object = _EMBED_INHERIT,
) -> TargetFactory:
    """A ``TargetFactory`` for the ASB target (concurrency locked to 1).

    The default ``model`` matches ASB's de-facto GPT model (``gpt-4o-mini``,
    hardcoded in upstream's memory-db path and refusal judge). The target runs
    inference through its OWN litellm proxy client, so the target model is a
    construction concern and is NOT subject to the optimizer's ``LLMClient``
    model-lock or token budget (those apply only to the attacker).

    ASB uses a process-global request queue, a singleton scheduler, and other
    process globals, so ``concurrency=1`` serializes tasks within one Controller
    AND only one ASB Controller may run per process: sweep multiple ASB threat
    models sequentially, not via a concurrent ``asyncio.gather`` of ASB
    Controllers (see the target's ASSUMPTIONS G.1). The memory store embeds
    through the same proxy (``embed_model``).
    """
    return TargetFactory(
        create=lambda: AsbTarget(
            model=model,
            api_base=api_base,
            api_key=api_key,
            request_delay_seconds=request_delay_seconds,
            max_output_tokens=max_output_tokens,
            embed_model=embed_model,
            embed_api_base=embed_api_base,
            embed_api_key=embed_api_key,
        ),
        concurrency=1,
    )


__all__ = [
    "asb_claim",
    "asb_agent_claim",
    "asb_full_claim",
    "asb_aggressive_claim",
    "asb_non_aggressive_claim",
    "asb_stealthy_claim",
    "asb_disruptive_claim",
    "asb_combined_claim",
    "asb_target_factory",
]
