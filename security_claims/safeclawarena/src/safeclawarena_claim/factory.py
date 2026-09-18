"""Factories for the SafeClawArena claim.

``safeclawarena_claim(...)`` builds a :class:`~superred.core.interfaces.security_claim.SecurityClaim`
of one :class:`~safeclawarena_claim.task.SafeClawArenaTask` per selected task,
with filters by dimension / category / difficulty / task-id. Convenience
per-dimension builders wrap it. ``safeclawarena_target_factory(...)`` returns a
:class:`~superred.core.controller.TargetFactory` for pairing.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from safeclawarena_target import SafeClawArenaTarget, safeclawarena_target_factory
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from safeclawarena_claim.categories import DIMENSIONS, load_tasks
from safeclawarena_claim.task import SafeClawArenaTask


def safeclawarena_claim(
    *,
    dimensions: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    difficulties: Iterable[str] | None = None,
    task_ids: Iterable[str] | None = None,
    platform: str = "openclaw",
) -> SecurityClaim[SafeClawArenaTarget]:
    """Build a SafeClawArena claim.

    Args:
        dimensions: restrict to these dimensions (``"ipi"``/``"pse"``/``"ssi"``/``"cdf"``);
            default all four (406 tasks).
        categories: restrict to these ``metadata.category`` values (e.g. ``"4.5"``).
        difficulties: restrict to these ``metadata.difficulty`` values.
        task_ids: restrict to these exact task ids (exclusive filter).
        platform: container replica for every task (``"openclaw"`` default).
    """
    dims = tuple(d.lower() for d in dimensions) if dimensions is not None else None
    if dims is not None:
        unknown = set(dims) - set(DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown dimensions {sorted(unknown)}; known: {list(DIMENSIONS)}")
    cat_set = {str(c) for c in categories} if categories is not None else None
    diff_set = {str(d) for d in difficulties} if difficulties is not None else None
    id_set = {str(t) for t in task_ids} if task_ids is not None else None

    tasks: list[SafeClawArenaTask] = []
    for task in load_tasks(dims):
        meta = task["metadata"]
        if cat_set is not None and str(meta.get("category")) not in cat_set:
            continue
        if diff_set is not None and str(meta.get("difficulty", "intermediate")) not in diff_set:
            continue
        if id_set is not None and str(meta["task_id"]) not in id_set:
            continue
        tasks.append(SafeClawArenaTask(task=task, platform=platform))

    if id_set is not None:
        found = {t.task_id for t in tasks}
        missing = id_set - found
        if missing:
            raise ValueError(f"unknown task_ids: {sorted(missing)}")

    return SecurityClaim.from_tasks(cast("list[Task[SafeClawArenaTarget]]", tasks))


def safeclawarena_ipi_claim(**kwargs: Any) -> SecurityClaim[SafeClawArenaTarget]:
    """Indirect-Prompt-Injection tasks (100)."""
    return safeclawarena_claim(dimensions=("ipi",), **kwargs)


def safeclawarena_pse_claim(**kwargs: Any) -> SecurityClaim[SafeClawArenaTarget]:
    """Persistent-State-Exploitation tasks (60)."""
    return safeclawarena_claim(dimensions=("pse",), **kwargs)


def safeclawarena_ssi_claim(**kwargs: Any) -> SecurityClaim[SafeClawArenaTarget]:
    """Skill-Supply-chain-Integrity tasks (100)."""
    return safeclawarena_claim(dimensions=("ssi",), **kwargs)


def safeclawarena_cdf_claim(**kwargs: Any) -> SecurityClaim[SafeClawArenaTarget]:
    """Cross-boundary-Data-Flow tasks (146)."""
    return safeclawarena_claim(dimensions=("cdf",), **kwargs)


# The paired target factory is re-exported from ``safeclawarena_target`` so a
# caller can build both halves from one import.
__all__ = [
    "safeclawarena_claim",
    "safeclawarena_ipi_claim",
    "safeclawarena_pse_claim",
    "safeclawarena_ssi_claim",
    "safeclawarena_cdf_claim",
    "safeclawarena_target_factory",
]
