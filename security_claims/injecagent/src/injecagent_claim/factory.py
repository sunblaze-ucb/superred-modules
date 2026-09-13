"""Factories for the InjecAgent claim.

``injecagent_claim(...)`` builds a
:class:`~superred.core.interfaces.security_claim.SecurityClaim` of one
:class:`~injecagent_claim.task.InjecAgentTask` per selected test case, filtered
by setting (base / enhanced), attack type (dh / ds), and task id. Convenience
builders wrap the common slices.

Pair with the target from the ``injecagent_target`` package::

    from injecagent_claim import injecagent_claim
    from injecagent_target import injecagent_target_factory
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.target import Target

from injecagent_claim.categories import ATTACKS, load_cases
from injecagent_claim.task import InjecAgentTask


def injecagent_claim(
    *,
    settings: Iterable[str] | None = None,
    attacks: Iterable[str] | None = None,
    task_ids: Iterable[str] | None = None,
) -> SecurityClaim[Target]:
    """Build an InjecAgent claim.

    Args:
        settings: restrict to these settings (``"base"`` and/or ``"enhanced"``);
            default ``("base",)`` — the paper's 1,054-case headline corpus.
        attacks: restrict to these attack types (``"dh"`` and/or ``"ds"``);
            default both.
        task_ids: restrict to these exact synthesized ids (exclusive filter).
    """
    settings_t = tuple(settings) if settings is not None else ("base",)
    attacks_t = tuple(attacks) if attacks is not None else ATTACKS
    id_set = {str(t) for t in task_ids} if task_ids is not None else None

    tasks: list[InjecAgentTask] = []
    for case in load_cases(settings=settings_t, attacks=attacks_t):
        if id_set is not None and case["_task_id"] not in id_set:
            continue
        tasks.append(InjecAgentTask(case=case))

    if id_set is not None:
        found = {t.task_id for t in tasks}
        missing = id_set - found
        if missing:
            raise ValueError(f"unknown task_ids: {sorted(missing)}")

    return SecurityClaim.from_tasks(tasks)


def injecagent_dh_claim(**kwargs: Any) -> SecurityClaim[Target]:
    """Direct-Harm tasks (510 per setting)."""
    return injecagent_claim(attacks=("dh",), **kwargs)


def injecagent_ds_claim(**kwargs: Any) -> SecurityClaim[Target]:
    """Data-Stealing tasks (544 per setting)."""
    return injecagent_claim(attacks=("ds",), **kwargs)


def injecagent_enhanced_claim(**kwargs: Any) -> SecurityClaim[Target]:
    """The ``enhanced`` setting (injection reinforced with a hacking prompt)."""
    return injecagent_claim(settings=("enhanced",), **kwargs)


__all__ = [
    "injecagent_claim",
    "injecagent_dh_claim",
    "injecagent_ds_claim",
    "injecagent_enhanced_claim",
]
