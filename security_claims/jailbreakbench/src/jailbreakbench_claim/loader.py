"""Load the vendored JBB-Behaviors harmful split.

Upstream data: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
(MIT), ``data/harmful-behaviors.csv`` — 100 behaviors, columns
``Index, Goal, Target, Behavior, Category, Source``.
"""

from __future__ import annotations

import csv
from importlib import resources

_DATA_FILES = {
    "harmful": "harmful-behaviors.csv",
    "benign": "benign-behaviors.csv",
}
#: ``harmful`` is the attack set. ``benign`` is upstream's paired *control*
#: set: 100 innocuous requests used to measure over-refusal, not jailbreaks.
DATASETS: tuple[str, ...] = ("harmful", "benign")
_DATA_FILE = _DATA_FILES["harmful"]
EXPECTED_COLUMNS = ("Index", "Goal", "Target", "Behavior", "Category", "Source")


def load_behaviors(
    *,
    dataset: str = "harmful",
    categories: tuple[str, ...] | list[str] | None = None,
    sources: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Return the behaviours, optionally filtered.

    Args:
        dataset: ``"harmful"`` (the attack set, default) or ``"benign"``
            (upstream's paired control set of innocuous requests, used to
            measure over-refusal -- see ASSUMPTIONS.md).
        categories: Keep only these ``Category`` values (case-insensitive).
        sources: Keep only these ``Source`` values (case-insensitive).
        limit: Keep at most this many rows, after filtering, in file order.

    Returns:
        One dict per behaviour, with the upstream columns verbatim.

    Raises:
        ValueError: on an unknown column set, an empty file, an out-of-range
            ``limit``, or a filter that matches nothing.
    """
    if dataset not in _DATA_FILES:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {list(DATASETS)}"
        )
    file_name = _DATA_FILES[dataset]
    text = (
        resources.files("jailbreakbench_claim.data")
        .joinpath(file_name)
        .read_text(encoding="utf-8")
    )
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f"{file_name} is empty")
    if tuple(rows[0].keys()) != EXPECTED_COLUMNS:
        raise ValueError(
            f"unexpected columns {tuple(rows[0].keys())!r}; "
            f"expected {EXPECTED_COLUMNS!r}"
        )

    if categories is not None:
        wanted = {c.lower() for c in categories}
        rows = [r for r in rows if r["Category"].lower() in wanted]
    if sources is not None:
        wanted = {s.lower() for s in sources}
        rows = [r for r in rows if r["Source"].lower() in wanted]
    if not rows:
        raise ValueError("no behaviours matched the given filters")

    if limit is not None:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit!r}")
        rows = rows[:limit]

    return rows


def all_categories() -> list[str]:
    """Every distinct ``Category`` in the vendored data, sorted."""
    return sorted({r["Category"] for r in load_behaviors()})
