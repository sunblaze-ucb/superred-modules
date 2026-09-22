"""Factory for the Aegis :class:`SecurityClaim`.

``aegis_claim(...)`` is the user-facing entry point. It loads the vendored Aegis
1.0 **test split**, keeps the unsafe **user-message** prompts, and produces one
:class:`aegis_claim.task.AegisTask` per surviving row, bundled into a
:class:`SecurityClaim`.

The data is vendored (public, CC-BY-4.0), so there is no runtime download — the
factory is offline. Only the judge makes network calls, at evaluation time.
"""

from __future__ import annotations

import logging
from typing import Final, cast

from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.target import Target
from superred.core.interfaces.task import Task
from superred.core.types.llm import LLMConfig

from aegis_claim.categories import atomic_categories
from aegis_claim.judge import DEFAULT_JUDGE_MODEL, AegisSafetyJudge, Judge
from aegis_claim.loader import load_unsafe_user_prompts
from aegis_claim.task import AegisTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provenance (pinned; see scripts/sync_upstream.py and _vendor/SHA256SUMS).
# The Aegis 1.0 repo is public (gated=False) and CC-BY-4.0.
# ---------------------------------------------------------------------------
DATASET_REPO_ID: Final = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"
DATASET_REVISION: Final = "bd96d862068e47630197de64eb91f8d1481ff3e0"
DATASET_SPLIT: Final = "test"

#: Unsafe user-message counts on the pinned test split (asserted in tests).
UNSAFE_COUNT_WITH_NEEDS_CAUTION: Final = 236
UNSAFE_COUNT_HARM_ONLY: Final = 180


def aegis_claim(
    *,
    judge: Judge | None = None,
    judge_llm_config: LLMConfig | None = None,
    include_needs_caution: bool = True,
    categories: list[str] | None = None,
    limit: int | None = None,
) -> SecurityClaim[Target]:
    """Build a :class:`SecurityClaim` evaluating Aegis against a chatbot target.

    Args:
        judge: A pluggable :class:`~aegis_claim.judge.Judge`. If ``None``, an
            :class:`~aegis_claim.judge.AegisSafetyJudge` is built from
            ``judge_llm_config``.
        judge_llm_config: Required if ``judge`` is ``None``. The judge uses its
            own :class:`LLMClient`, so judge cost is not tracked in the
            controller's ``RunResult.llm_usage``; it surfaces in the rationale.
        include_needs_caution: When True (default, deepteam-faithful
            ``labels_0 != "Safe"`` filter), prompts labelled only
            ``"Needs Caution"`` are included (236 tasks on the pinned split).
            When False, only the 13 harm categories count (180 tasks). See
            ``ASSUMPTIONS.md``.
        categories: Optional list of Aegis risk-category names (dataset
            spellings, e.g. ``["Violence", "Hate/Identity Hate"]``) to keep. A
            row survives if any of its atomic labels is in this set. ``None``
            keeps all categories.
        limit: Optional cap on the number of tasks (first ``limit`` in file
            order), for cheaper pilot runs. ``None`` keeps all.

    Returns:
        A :class:`SecurityClaim` bundling one :class:`AegisTask` per surviving
        row.

    Raises:
        ValueError: If neither ``judge`` nor ``judge_llm_config`` is provided,
            if ``limit`` is negative, or if the filters leave no tasks.
    """
    if judge is None:
        if judge_llm_config is None:
            raise ValueError(
                "Must provide either `judge` or `judge_llm_config`. Pass "
                f"`judge_llm_config=LLMConfig(model={DEFAULT_JUDGE_MODEL!r}, ...)` "
                "for the default Aegis safety judge."
            )
        judge = AegisSafetyJudge.from_config(judge_llm_config)

    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")

    rows = load_unsafe_user_prompts(include_needs_caution=include_needs_caution)

    if categories is not None:
        wanted = set(categories)
        rows = [
            row
            for row in rows
            if any(cat in wanted for cat in atomic_categories(row["labels_0"]))
        ]

    if limit is not None:
        rows = rows[:limit]

    tasks: list[AegisTask] = [
        AegisTask(
            prompt=row["text"],
            risk_category=row["labels_0"],
            row_id=row["id"],
            judge=judge,
        )
        for row in rows
    ]

    if not tasks:
        raise ValueError(
            "No tasks produced. Check the `categories` / `limit` / "
            "`include_needs_caution` filters are not over-restrictive."
        )

    # Cast widens SecurityClaim[ChatbotTarget] to SecurityClaim[Target] so the
    # Controller (which takes SecurityClaim[Target]) accepts it; the underlying
    # tasks remain Task[ChatbotTarget].
    return SecurityClaim.from_tasks(cast(list[Task[Target]], tasks))


__all__ = [
    "aegis_claim",
    "DATASET_REPO_ID",
    "DATASET_REVISION",
    "DATASET_SPLIT",
    "UNSAFE_COUNT_WITH_NEEDS_CAUTION",
    "UNSAFE_COUNT_HARM_ONLY",
]
