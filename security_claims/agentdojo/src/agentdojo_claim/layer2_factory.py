"""Layer-2 factory: build SecurityClaim instances from the goal catalogue.

Two main factories:

- :func:`agentdojo_layer2_claim` — top: all Layer-2 goals, with
  optional filtering by goal_id or category.
- :func:`agentdojo_layer2_category_claim` — per-category roll-up.

The v1 catalogue ships 19 goals (4 starters, 9 per-suite expansions,
4 cross-suite, 2 capability-misuse); extending it
is a matter of adding new ``Layer2GoalSpec`` modules under
``layer2_goals/`` and appending them to :data:`GOAL_SPECS`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from agentdojo_target.target import AgentDojoTarget
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.target import Target
from superred.core.interfaces.task import Task

from agentdojo_claim.layer2_goals import GOAL_SPECS
from agentdojo_claim.layer2_task import (
    Layer2GoalSpec,
    SystemViolatingTask,
)


def _build_tasks(specs: Iterable[Layer2GoalSpec]) -> list[SystemViolatingTask]:
    return [SystemViolatingTask(s) for s in specs]


def agentdojo_layer2_claim(
    *,
    goal_ids: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
) -> SecurityClaim[AgentDojoTarget]:
    """Build the Layer-2 :class:`SecurityClaim`.

    Args:
        goal_ids: Iterable of ``Layer2GoalSpec.goal_id`` strings.  If
            provided, only matching goals are returned.
        categories: Iterable of category labels to keep.

    Raises:
        ValueError: If a referenced goal_id or category is unknown.
    """
    selected = list(GOAL_SPECS)
    if goal_ids is not None:
        wanted_ids = set(goal_ids)
        all_ids = {s.goal_id for s in selected}
        unknown = wanted_ids - all_ids
        if unknown:
            raise ValueError(f"Unknown Layer-2 goal_id(s): {sorted(unknown)}")
        selected = [s for s in selected if s.goal_id in wanted_ids]
    if categories is not None:
        wanted_cats = set(categories)
        all_cats = {s.category for s in selected}
        unknown_cats = wanted_cats - all_cats
        if unknown_cats:
            raise ValueError(f"Unknown Layer-2 category(s): {sorted(unknown_cats)}")
        selected = [s for s in selected if s.category in wanted_cats]
    tasks = _build_tasks(selected)
    return SecurityClaim.from_tasks(cast(list[Task[Target]], tasks))


def agentdojo_layer2_category_claim(category: str) -> SecurityClaim[AgentDojoTarget]:
    """Per-category roll-up."""
    return agentdojo_layer2_claim(categories=[category])


def layer2_categories() -> frozenset[str]:
    """Return the set of category labels present in the v1 catalogue."""
    return frozenset(s.category for s in GOAL_SPECS)


def layer2_goal_ids() -> frozenset[str]:
    """Return the set of goal_ids present in the v1 catalogue."""
    return frozenset(s.goal_id for s in GOAL_SPECS)


__all__ = [
    "agentdojo_layer2_claim",
    "agentdojo_layer2_category_claim",
    "layer2_categories",
    "layer2_goal_ids",
]
