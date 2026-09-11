"""Factories for the HackAPrompt claim.

``hackaprompt_claim(...)`` builds a
:class:`~superred.core.interfaces.security_claim.SecurityClaim` of one
:class:`~hackaprompt_claim.task.HackAPromptTask` per selected level (0-10).
``hackaprompt_target_factory`` is re-exported so a caller can build both halves
from one import.
"""

from __future__ import annotations

from collections.abc import Iterable

from hackaprompt_target import HackAPromptTarget, hackaprompt_target_factory
from superred.core.interfaces.security_claim import SecurityClaim

from hackaprompt_claim.categories import LEVELS
from hackaprompt_claim.task import HackAPromptTask


def hackaprompt_claim(
    *, levels: Iterable[int] | None = None
) -> SecurityClaim[HackAPromptTarget]:
    """Build a HackAPrompt claim.

    Args:
        levels: restrict to these levels (0-10); default all 11.
    """
    chosen = tuple(levels) if levels is not None else LEVELS
    unknown = [n for n in chosen if n not in LEVELS]
    if unknown:
        raise ValueError(f"unknown levels {unknown}; expected 0-10")
    tasks = [HackAPromptTask(level=n) for n in chosen]
    return SecurityClaim.from_tasks(tasks)


__all__ = ["hackaprompt_claim", "hackaprompt_target_factory"]
