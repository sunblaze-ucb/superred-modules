"""Factory functions building SafeClawBench Exec-Balanced claims.

``safeclawbench_exec_claim(...)`` enumerates one :class:`SafeClawBenchExecTask`
per selected executable scenario (family / id / limit filters). Convenience
roll-ups cover per-family and combined axes.
``safeclawbench_exec_target_factory(...)`` (re-exported) wires the sandbox target
for these claims.

Unlike the Semantic Core claim, scoring is a **deterministic state oracle** — no
judge LLM is required (only the model-under-test, supplied to the target).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from safeclawbench_exec_target import (
    ATTACK_FAMILIES,
    SafeClawBenchExecTarget,
    load_scenarios,
    safeclawbench_exec_target_factory,
)
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from safeclawbench_exec_claim.task import SafeClawBenchExecTask


def safeclawbench_exec_claim(
    *,
    attack_types: Iterable[str] | None = None,
    task_ids: Iterable[str] | None = None,
    limit: int | None = None,
    defense_level: str | None = None,
) -> SecurityClaim[SafeClawBenchExecTarget]:
    """Build a SafeClawBench Exec-Balanced (sandbox-harm) claim.

    Args:
        attack_types: restrict to these reporting families (DPI/IPI/TRI/MPI/MEX/
            ADI); default all six.
        task_ids: restrict to these explicit case ids (exclusive filter).
        limit: keep at most this many cases (after filtering, file order).
        defense_level: prompt-level defense policy applied by the target
            (``D0``..``D3``); default ``None`` leaves the target's own default.
    """
    fam_set = set(attack_types) if attack_types is not None else None
    id_set = set(task_ids) if task_ids is not None else None

    if fam_set is not None:
        unknown = fam_set - set(ATTACK_FAMILIES)
        if unknown:
            raise ValueError(
                f"Unknown attack_types: {sorted(unknown)}; known: {list(ATTACK_FAMILIES)}"
            )

    selected = []
    for scenario in load_scenarios():
        if fam_set is not None and scenario.attack_type not in fam_set:
            continue
        if id_set is not None and scenario.case_id not in id_set:
            continue
        selected.append(scenario)

    if limit is not None:
        selected = selected[:limit]

    tasks = [
        SafeClawBenchExecTask(scenario, defense_level=defense_level)
        for scenario in selected
    ]
    if not tasks:
        raise ValueError(
            "safeclawbench_exec_claim produced no tasks for the given filters"
        )
    return SecurityClaim.from_tasks(
        cast("list[Task[SafeClawBenchExecTarget]]", tasks)
    )


def safeclawbench_exec_family_claim(
    family: str, **kwargs: Any
) -> SecurityClaim[SafeClawBenchExecTarget]:
    """All executable cases for a single reporting family (e.g. ``"MPI"``)."""
    return safeclawbench_exec_claim(attack_types=[family], **kwargs)


def safeclawbench_exec_combined_claim(
    claims: list[SecurityClaim[Any]],
) -> SecurityClaim[Any]:
    """Compose several Exec-Balanced sub-claims into one (lazy chaining)."""
    return SecurityClaim.from_claims(claims)


__all__ = [
    "safeclawbench_exec_claim",
    "safeclawbench_exec_combined_claim",
    "safeclawbench_exec_family_claim",
    "safeclawbench_exec_target_factory",
]
