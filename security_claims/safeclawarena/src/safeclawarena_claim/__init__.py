"""safeclawarena_claim: the SafeClawArena benchmark claim for the SafeClawArena target.

One Task per SafeClawArena adversarial task (406 across SSI / PSE / CDF / IPI);
upstream's deterministic canary/taint check logic and dimension scoring ported
verbatim into a pure judge; hierarchical factories by dimension / category /
difficulty. Pairs with the ``safeclawarena_target`` package (the container
platform replica + Sim-Google environment + post-run state capture).
"""

from __future__ import annotations

from safeclawarena_claim.categories import (
    DIMENSION_NAMES,
    DIMENSIONS,
    load_tasks,
    task_counts,
)
from safeclawarena_claim.factory import (
    safeclawarena_cdf_claim,
    safeclawarena_claim,
    safeclawarena_ipi_claim,
    safeclawarena_pse_claim,
    safeclawarena_ssi_claim,
    safeclawarena_target_factory,
)
from safeclawarena_claim.judge import (
    CheckResult,
    PostState,
    TaskVerdict,
    evaluate_check,
    evaluate_detection,
    evaluate_taint_leakage,
    evaluate_task,
)
from safeclawarena_claim.task import SafeClawArenaTask

__all__ = [
    "DIMENSIONS",
    "DIMENSION_NAMES",
    "load_tasks",
    "task_counts",
    "SafeClawArenaTask",
    "safeclawarena_claim",
    "safeclawarena_ipi_claim",
    "safeclawarena_pse_claim",
    "safeclawarena_ssi_claim",
    "safeclawarena_cdf_claim",
    "safeclawarena_target_factory",
    "PostState",
    "CheckResult",
    "TaskVerdict",
    "evaluate_check",
    "evaluate_detection",
    "evaluate_taint_leakage",
    "evaluate_task",
]
