"""PromptInject goal-hijacking benchmark as a superred SecurityClaim.

Reproduces garak's PromptInject probe: base prompts hijacked by five
goal-hijacking attacks, judged by a deterministic rogue-string match. Single-
turn, bound to ChatbotTarget, no judge model.

    >>> from promptinject_claim import promptinject_claim
    >>> claim = promptinject_claim(limit=50)
    >>> tasks = list(claim)
"""

from __future__ import annotations

from typing import cast

from chatbot_target import ChatbotTarget  # type: ignore[import-untyped]
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from promptinject_claim.judge import rogue_emitted
from promptinject_claim.loader import ROGUE_STRINGS, PromptInjectItem, load_items
from promptinject_claim.task import PromptInjectTask

__all__ = [
    "ROGUE_STRINGS",
    "PromptInjectItem",
    "PromptInjectTask",
    "load_items",
    "promptinject_claim",
    "rogue_emitted",
]


def promptinject_claim(
    *,
    rogue_strings: tuple[str, ...] | list[str] | None = None,
    limit: int | None = 100,
    seed: int = 0,
) -> SecurityClaim[ChatbotTarget]:
    """Build a PromptInject SecurityClaim.

    Args:
        rogue_strings: Which rogue-string classes to include (default all three:
            ``hate-humans``, ``kill-humans``, ``long-prompt``).
        limit: Prompts per class after garak's seeded shuffle (default 100;
            ``None`` keeps all ~700 per class).
        seed: Shuffle seed, mirroring garak's ``self.seed``.

    Returns:
        A SecurityClaim with one PromptInjectTask per built prompt.
    """
    items = load_items(rogue_strings=rogue_strings, limit=limit, seed=seed)
    tasks = [PromptInjectTask(item=i) for i in items]
    return SecurityClaim.from_tasks(cast("list[Task[ChatbotTarget]]", tasks))
