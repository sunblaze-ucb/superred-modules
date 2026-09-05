"""Snowball hallucination benchmark as a superred SecurityClaim.

Three question sets from garak (graph connectivity, primality, senator search),
each judged by a deterministic regex detector — a hallucinated confident answer
counts as a success. Single-turn, bound to ChatbotTarget, no judge model.

    >>> from snowball_claim import snowball_claim
    >>> claim = snowball_claim(limit=100)
    >>> tasks = list(claim)
"""

from __future__ import annotations

from typing import cast

from chatbot_target import ChatbotTarget  # type: ignore[import-untyped]
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from snowball_claim.judge import hallucinated
from snowball_claim.loader import SUBSET_NAMES, SnowballItem, load_items
from snowball_claim.task import SnowballTask

__all__ = [
    "SUBSET_NAMES",
    "SnowballItem",
    "SnowballTask",
    "hallucinated",
    "load_items",
    "snowball_claim",
]


def snowball_claim(
    *,
    subsets: tuple[str, ...] | list[str] | None = None,
    limit: int | None = 100,
) -> SecurityClaim[ChatbotTarget]:
    """Build a Snowball hallucination SecurityClaim.

    Args:
        subsets: Which question sets to include (default all three:
            ``graph_connectivity``, ``primes``, ``senators``).
        limit: Questions per subset, from the end of the file — ``100``
            matches garak's default-active capped variants; ``None`` uses all
            500 per subset (garak's ``*Full`` variants).

    Returns:
        A SecurityClaim with one SnowballTask per selected question.
    """
    items = load_items(subsets=subsets, limit=limit)
    tasks = [SnowballTask(item=i) for i in items]
    return SecurityClaim.from_tasks(cast("list[Task[ChatbotTarget]]", tasks))
