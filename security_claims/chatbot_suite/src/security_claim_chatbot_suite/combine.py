"""Generic, claim-agnostic combinator for superred SecurityClaims.

``combine_claims`` unions the tasks of several ``SecurityClaim``s into one,
removing functional duplicates (tasks whose normalized goal text matches an
already-kept task) so the combined claim never spends attacker/judge tokens
re-testing the same harmful behaviour. An optional per-category cap takes a
stratified subset across each source's native category taxonomy.

The combinator is deliberately benchmark-agnostic: it only touches the
``Task`` ABC surface (``task.goal.description``) plus a pluggable
``category_getter``. Each surviving task keeps its own native judge and
configuration; there is no shared judge.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

logger = logging.getLogger(__name__)

# Attribute names benchmark tasks use to expose their category, tried in
# order. HarmBench -> ``semantic_category``; StrongREJECT -> ``category``;
# SORRY-Bench -> ``category_name``. Namespaced by attribute name in the
# returned key so categories never collide across benchmarks.
_CATEGORY_ATTRS = ("semantic_category", "category", "category_name")


def normalize_goal(text: str) -> str:
    """Whitespace-collapsed, case-folded form used for duplicate detection.

    Catches exact and trivial-variant duplicates (case, surrounding/internal
    whitespace) deterministically and cheaply. Semantic near-duplicates are
    out of scope (they would need embeddings or an LLM, adding cost and
    nondeterminism).
    """
    return " ".join(text.split()).strip().casefold()


def category_of(task: Task) -> str:
    """Best-effort category key for a heterogeneous benchmark task.

    Returns ``"<attr>=<value>"`` for the first populated category attribute,
    so HarmBench/StrongREJECT/SORRY-Bench categories live in disjoint
    namespaces. Falls back to ``"_uncategorized"``.
    """
    for attr in _CATEGORY_ATTRS:
        value = getattr(task, attr, None)
        if value is not None and str(value).strip():
            return f"{attr}={value}"
    return "_uncategorized"


@dataclass(frozen=True)
class TaskRecord:
    """One surviving task's provenance, for an external analysis manifest."""

    index: int  # 1-based position in the combined claim (== persisted file index)
    source: str  # source claim label
    category: str  # category key from ``category_getter``
    task_class: str  # concrete Task subclass name
    goal_preview: str  # first 120 chars of the goal text


@dataclass
class CombineStats:
    """Summary of a combine pass (for logging / provenance)."""

    total_input: int = 0
    kept: int = 0
    dropped_duplicate: int = 0
    dropped_over_cap: int = 0
    kept_per_source: dict[str, int] = field(default_factory=dict)
    manifest: list[TaskRecord] = field(default_factory=list)


def combine_claims(
    sources: Iterable[tuple[str, SecurityClaim]],
    *,
    dedup: bool = True,
    normalizer: Callable[[str], str] = normalize_goal,
    max_per_category: int | None = None,
    category_getter: Callable[[Task], str] = category_of,
    manifest_out: list[TaskRecord] | None = None,
    stats_out: list[CombineStats] | None = None,
) -> SecurityClaim:
    """Union labelled source claims into one deduplicated SecurityClaim.

    Single pass, first-seen wins, order preserved. A task is dropped if its
    normalized goal already appeared (when ``dedup``) or its category is
    already at ``max_per_category``. The first source listed wins any
    cross-source duplicate, so list sources in priority order.

    Args:
        sources: ``(label, claim)`` pairs. The label tags the task's
            provenance in the manifest.
        dedup: drop tasks whose normalized goal text was already kept.
        normalizer: goal-text -> dedup key.
        max_per_category: keep at most this many tasks per category key
            (``None`` = no cap).
        category_getter: task -> category key.
        manifest_out: if given, ``TaskRecord``s are appended here (kept order).
        stats_out: if given, the ``CombineStats`` is appended here.

    Returns:
        ``SecurityClaim.from_tasks(survivors)`` (non-empty, order preserved).

    Raises:
        ValueError: if no task survives.
    """
    seen_keys: set[str] = set()
    category_counts: dict[str, int] = {}
    survivors: list[Task] = []
    stats = CombineStats()

    for source_name, claim in sources:
        for task in claim:
            stats.total_input += 1
            key = normalizer(task.goal.description)
            if dedup and key in seen_keys:
                stats.dropped_duplicate += 1
                continue
            category = category_getter(task)
            if (
                max_per_category is not None
                and category_counts.get(category, 0) >= max_per_category
            ):
                stats.dropped_over_cap += 1
                continue

            seen_keys.add(key)
            category_counts[category] = category_counts.get(category, 0) + 1
            survivors.append(task)
            stats.kept += 1
            stats.kept_per_source[source_name] = (
                stats.kept_per_source.get(source_name, 0) + 1
            )
            record = TaskRecord(
                index=len(survivors),
                source=source_name,
                category=category,
                task_class=type(task).__name__,
                goal_preview=task.goal.description[:120],
            )
            stats.manifest.append(record)
            if manifest_out is not None:
                manifest_out.append(record)

    if not survivors:
        raise ValueError(
            "combine_claims produced no tasks (all sources empty or fully deduped)"
        )

    logger.info(
        "combine_claims: kept %d of %d tasks "
        "(%d duplicates, %d over per-category cap); per-source: %s",
        stats.kept,
        stats.total_input,
        stats.dropped_duplicate,
        stats.dropped_over_cap,
        stats.kept_per_source,
    )

    if stats_out is not None:
        stats_out.append(stats)

    return SecurityClaim.from_tasks(survivors)
