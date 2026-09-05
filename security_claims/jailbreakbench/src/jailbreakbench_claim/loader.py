"""Load the vendored JBB-Behaviors harmful split.

Upstream data: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
(MIT), ``data/harmful-behaviors.csv`` — 100 behaviors, columns
``Index, Goal, Target, Behavior, Category, Source``.
"""

from __future__ import annotations

import csv
from importlib import resources

_DATA_FILE = "harmful-behaviors.csv"
EXPECTED_COLUMNS = ("Index", "Goal", "Target", "Behavior", "Category", "Source")


def load_behaviors(
    *,
    categories: tuple[str, ...] | list[str] | None = None,
    sources: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Return the harmful behaviours, optionally filtered.

    Args:
        categories: Keep only these ``Category`` values (case-insensitive).
        sources: Keep only these ``Source`` values (case-insensitive).
        limit: Keep at most this many rows, after filtering, in file order.

    Returns:
        One dict per behaviour, with the upstream columns verbatim.

    Raises:
        ValueError: on an unknown column set, an empty file, an out-of-range
            ``limit``, or a filter that matches nothing.
    """
    text = (
        resources.files("jailbreakbench_claim.data")
        .joinpath(_DATA_FILE)
        .read_text(encoding="utf-8")
    )
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError("harmful-behaviors.csv is empty")
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
