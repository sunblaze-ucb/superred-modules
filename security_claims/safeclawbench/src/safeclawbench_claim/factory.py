"""Factory functions building SafeClawBench Semantic Core claims.

``safeclawbench_claim(...)`` enumerates one :class:`SafeClawBenchTask` per
selected Semantic Core case, with family/harm/difficulty/id filters.
Convenience roll-ups cover the natural axes (per attack family, combined).
``safeclawbench_target_factory(...)`` wires an OpenClaw target for these
claims.

CoreFail is an LLM-judged endpoint, so a judge is required: pass either a
pre-built ``judge`` (e.g. for tests) or a ``judge_llm_config``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from openclaw_target import OpenClawTarget, openclaw_target_factory
from superred.core.controller import TargetFactory
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task
from superred.core.types.llm import LLMConfig

from safeclawbench_claim.judge import CoreFailJudge, LLMCoreFailJudge
from safeclawbench_claim.loader import ATTACK_FAMILIES, load_cases
from safeclawbench_claim.task import SafeClawBenchTask


def _build_judge(
    judge: CoreFailJudge | None, judge_llm_config: LLMConfig | None
) -> CoreFailJudge:
    if judge is not None:
        return judge
    if judge_llm_config is not None:
        return LLMCoreFailJudge.from_config(judge_llm_config)
    raise ValueError(
        "safeclawbench_claim requires a CoreFail judge: pass judge=... or "
        "judge_llm_config=... (CoreFail@600 is an LLM-judged endpoint)."
    )


def safeclawbench_claim(
    *,
    attack_types: Iterable[str] | None = None,
    harm_types: Iterable[str] | None = None,
    difficulties: Iterable[str] | None = None,
    task_ids: Iterable[str] | None = None,
    exclude_seed: bool = False,
    limit: int | None = None,
    judge: CoreFailJudge | None = None,
    judge_llm_config: LLMConfig | None = None,
    defense_system_prompt: str | None = None,
) -> SecurityClaim[OpenClawTarget]:
    """Build a SafeClawBench Semantic Core claim.

    Args:
        attack_types: restrict to these reporting families (DPI/IPI/TRI/MPI/
            MEX/ADI); default all six.
        harm_types: restrict to these ``harm_type`` values (leak/action/persist/...).
        difficulties: restrict to these difficulty tags (easy/medium/hard).
        task_ids: restrict to these explicit case ids (exclusive filter).
        exclude_seed: drop the 8 cases flagged ``is_seed``.
        limit: keep at most this many cases (after filtering, file order).
        judge / judge_llm_config: the CoreFail judge (one is required).
        defense_system_prompt: optional text prepended to ``AGENT_SYSTEM_PROMPT``
            in the agent's system prompt. The default ``None`` matches upstream's
            D0 (no defense). It cannot express upstream's other policies, which
            also rewrite the user turn (see :class:`SafeClawBenchTask`).
    """
    judge_obj = _build_judge(judge, judge_llm_config)
    cases = load_cases()

    fam_set = set(attack_types) if attack_types is not None else None
    harm_set = set(harm_types) if harm_types is not None else None
    diff_set = set(difficulties) if difficulties is not None else None
    id_set = set(task_ids) if task_ids is not None else None

    if fam_set is not None:
        unknown = fam_set - set(ATTACK_FAMILIES)
        if unknown:
            raise ValueError(
                f"Unknown attack_types: {sorted(unknown)}; known: {list(ATTACK_FAMILIES)}"
            )

    selected = []
    for case in cases:
        if fam_set is not None and case.attack_type not in fam_set:
            continue
        if harm_set is not None and case.harm_type not in harm_set:
            continue
        if diff_set is not None and case.difficulty not in diff_set:
            continue
        if id_set is not None and case.task_id not in id_set:
            continue
        if exclude_seed and case.is_seed:
            continue
        selected.append(case)

    if limit is not None:
        selected = selected[:limit]

    tasks = [
        SafeClawBenchTask(
            case,
            judge_obj,
            defense_system_prompt=defense_system_prompt,
        )
        for case in selected
    ]
    if not tasks:
        raise ValueError("safeclawbench_claim produced no tasks for the given filters")
    return SecurityClaim.from_tasks(cast("list[Task[OpenClawTarget]]", tasks))


def safeclawbench_family_claim(
    family: str, **kwargs: Any
) -> SecurityClaim[OpenClawTarget]:
    """All cases for a single reporting family (e.g. ``"DPI"``)."""
    return safeclawbench_claim(attack_types=[family], **kwargs)


def safeclawbench_combined_claim(
    claims: list[SecurityClaim[Any]],
) -> SecurityClaim[Any]:
    """Compose several SafeClawBench sub-claims into one (lazy chaining)."""
    return SecurityClaim.from_claims(claims)


def safeclawbench_target_factory(
    *,
    enable_tool_injection: bool = False,
    enable_llm_proxy: bool = False,
    concurrency: int = 1,
    **target_kwargs: Any,
) -> TargetFactory:
    """A ``TargetFactory`` for the OpenClaw target wired for these claims.

    Thin delegation to :func:`openclaw_target.openclaw_target_factory`. The
    defaults leave tool-output injection and the LLM proxy off; enable them to
    drive the tool-output / system-prompt controllables. Either way the agent
    keeps its real OpenClaw tools, unlike upstream's Semantic Core, which makes
    no function calls (see the package README). Concurrency defaults to 1 for
    the local managed runtime (use the Docker runtime for parallel instances —
    see the OpenClaw target README).
    """
    return openclaw_target_factory(
        enable_tool_injection=enable_tool_injection,
        enable_llm_proxy=enable_llm_proxy,
        concurrency=concurrency,
        **target_kwargs,
    )


__all__ = [
    "safeclawbench_claim",
    "safeclawbench_combined_claim",
    "safeclawbench_family_claim",
    "safeclawbench_target_factory",
]
