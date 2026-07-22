"""Layer-1 factories: build :class:`SecurityClaim` instances at each level.

Three tiers of factories:

- :func:`agentdojo_layer1_claim` — top: all canonical pairs, or any
  caller-supplied subset.
- :func:`agentdojo_layer1_suite_claim` — per-suite roll-up of the
  canonical pairs.
- :func:`agentdojo_layer1_category_claim` — per-category roll-up.

Filtering: callers may pass ``pairs=``, ``suites=``, or
``categories=`` kwargs to ``agentdojo_layer1_claim`` to narrow the
default 27-pair scope.  Pairs that fail upstream lookup (e.g. an
unknown user_task_id) raise immediately so misconfiguration surfaces
fast.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from agentdojo_target.target import AgentDojoTarget
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.target import Target
from superred.core.interfaces.task import Task

from agentdojo_claim.layer1_bridge import get_injection_task, get_user_task
from agentdojo_claim.layer1_categories import (
    ALL_CATEGORIES,
    CATEGORIES_BY_SUITE,
    INJECTION_CATEGORIES,
    category_of,
)
from agentdojo_claim.layer1_pairs import CANONICAL_PAIRS, SUITES_IN_ORDER
from agentdojo_claim.layer1_task import AgentDojoPairedTask


def _build_task(suite: str, user_task_id: str, injection_task_id: str) -> AgentDojoPairedTask:
    """Construct one paired task; raises on unknown ids."""
    user_task = get_user_task(suite, user_task_id)
    injection_task = get_injection_task(suite, injection_task_id)
    category = category_of(suite, injection_task_id)
    return AgentDojoPairedTask(
        suite=suite,
        user_task_id=user_task_id,
        injection_task_id=injection_task_id,
        user_task=user_task,
        injection_task=injection_task,
        category=category,
    )


def agentdojo_layer1_claim(
    *,
    pairs: Iterable[tuple[str, str, str]] | None = None,
    suites: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
) -> SecurityClaim[AgentDojoTarget]:
    """Build the Layer-1 :class:`SecurityClaim`.

    Default: the 27 canonical pairs (:data:`CANONICAL_PAIRS`).

    Filter precedence (applied in order, narrowing each time):

    1. ``pairs`` (if provided): exclusive — uses *only* the listed
       pairs.  Bypasses both ``suites`` and ``categories``.  This is
       how the faithfulness-sweep scope passes the full 629-case
       cross-product.
    2. ``suites``: keep only canonical pairs whose suite matches.
    3. ``categories``: keep only canonical pairs whose injection-task
       category matches.

    Args:
        pairs: Iterable of (suite, user_task_id, injection_task_id)
            tuples.  ``None`` = use canonical 27-pair scope.
        suites: Iterable of suite names to keep (subset of
            ``{banking, workspace, slack, travel}``).
        categories: Iterable of category labels to keep
            (must be in :data:`ALL_CATEGORIES`).

    Returns:
        A :class:`SecurityClaim` lazily iterating the selected
        :class:`AgentDojoPairedTask` instances.

    Raises:
        ValueError: If a filter argument references an unknown name.
    """
    if pairs is not None:
        selected: list[tuple[str, str, str]] = list(pairs)
    else:
        selected = list(CANONICAL_PAIRS)
        if suites is not None:
            suites_set = set(suites)
            unknown = suites_set - set(SUITES_IN_ORDER)
            if unknown:
                raise ValueError(f"Unknown suite(s): {sorted(unknown)}")
            selected = [p for p in selected if p[0] in suites_set]
        if categories is not None:
            categories_set = set(categories)
            unknown_cats = categories_set - set(ALL_CATEGORIES)
            if unknown_cats:
                raise ValueError(f"Unknown category(s): {sorted(unknown_cats)}")
            selected = [
                p for p in selected
                if INJECTION_CATEGORIES.get((p[0], p[2])) in categories_set
            ]

    tasks: list[AgentDojoPairedTask] = []
    for suite, user_task_id, injection_task_id in selected:
        tasks.append(_build_task(suite, user_task_id, injection_task_id))

    # Widen variance for the Controller's SecurityClaim[Target] signature
    # (same pattern as sorry_bench_claim).
    return SecurityClaim.from_tasks(cast(list[Task[Target]], tasks))


def agentdojo_layer1_suite_claim(suite: str) -> SecurityClaim[AgentDojoTarget]:
    """Per-suite roll-up: all canonical pairs whose suite is *suite*."""
    if suite not in SUITES_IN_ORDER:
        raise ValueError(f"Unknown suite {suite!r}")
    return agentdojo_layer1_claim(suites=[suite])


def agentdojo_layer1_category_claim(category: str) -> SecurityClaim[AgentDojoTarget]:
    """Per-category roll-up: all canonical pairs whose injection task
    falls into *category*."""
    if category not in ALL_CATEGORIES:
        raise ValueError(f"Unknown category {category!r}")
    return agentdojo_layer1_claim(categories=[category])


# Convenience: pre-built per-suite + per-category factories so callers
# can import them by name (mirrors the SORRY-Bench pattern).
def agentdojo_layer1_banking_claim() -> SecurityClaim[AgentDojoTarget]:
    return agentdojo_layer1_suite_claim("banking")


def agentdojo_layer1_workspace_claim() -> SecurityClaim[AgentDojoTarget]:
    return agentdojo_layer1_suite_claim("workspace")


def agentdojo_layer1_slack_claim() -> SecurityClaim[AgentDojoTarget]:
    return agentdojo_layer1_suite_claim("slack")


def agentdojo_layer1_travel_claim() -> SecurityClaim[AgentDojoTarget]:
    return agentdojo_layer1_suite_claim("travel")


__all__ = [
    "agentdojo_layer1_claim",
    "agentdojo_layer1_suite_claim",
    "agentdojo_layer1_category_claim",
    "agentdojo_layer1_banking_claim",
    "agentdojo_layer1_workspace_claim",
    "agentdojo_layer1_slack_claim",
    "agentdojo_layer1_travel_claim",
]
