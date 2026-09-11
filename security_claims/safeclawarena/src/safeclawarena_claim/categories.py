"""Loaders and constants for the vendored SafeClawArena benchmark tasks.

The 406 adversarial tasks are vendored verbatim under ``data/tasks/<dim>/`` (MIT;
see ``data/SAFECLAWARENA_LICENSE``), one JSON per task, exactly as upstream ships
them. One source of truth for the factory and the tests.

Upstream: SafeClawArena, https://github.com/sunblaze-ucb/SafeClawArena
(SafeClawBench Authors, 2026). See this module's ``NOTICE`` and ``ASSUMPTIONS.md``.
"""

from __future__ import annotations

import json
import os
from typing import Any

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TASKS_DIR = os.path.join(_DATA_DIR, "tasks")

#: The four principle-aligned dimensions (upstream task-id prefixes / dir names).
DIMENSIONS: tuple[str, ...] = ("ipi", "pse", "ssi", "cdf")

#: Dimension code -> full name (from the schema / README).
DIMENSION_NAMES: dict[str, str] = {
    "ssi": "Skill Supply-Chain Integrity",
    "pse": "Persistent State Exploitation",
    "cdf": "Cross-Boundary Data Flow",
    "ipi": "Indirect Prompt Injection",
}


def load_tasks(
    dimensions: tuple[str, ...] | list[str] | None = None,
    tasks_dir: str = TASKS_DIR,
) -> list[dict[str, Any]]:
    """Load SafeClawArena task dicts, sorted by task_id for deterministic order.

    Args:
        dimensions: restrict to these dimension dirs (default: all four).
        tasks_dir: root of the vendored ``tasks/`` tree.
    """
    dims = tuple(dimensions) if dimensions is not None else DIMENSIONS
    unknown = set(dims) - set(DIMENSIONS)
    if unknown:
        raise ValueError(f"unknown dimensions {sorted(unknown)}; known: {list(DIMENSIONS)}")
    tasks: list[dict[str, Any]] = []
    for dim in dims:
        dim_dir = os.path.join(tasks_dir, dim)
        if not os.path.isdir(dim_dir):
            continue
        for name in sorted(os.listdir(dim_dir)):
            if name.endswith(".json"):
                with open(os.path.join(dim_dir, name), encoding="utf-8") as f:
                    tasks.append(json.load(f))
    tasks.sort(key=lambda t: t["metadata"]["task_id"])
    return tasks


def task_counts(tasks_dir: str = TASKS_DIR) -> dict[str, int]:
    """Return {dimension: n_tasks} for the vendored corpus."""
    return {dim: len(load_tasks((dim,), tasks_dir)) for dim in DIMENSIONS}


__all__ = ["DIMENSIONS", "DIMENSION_NAMES", "TASKS_DIR", "load_tasks", "task_counts"]
