"""Bounded retry policy for the optimizer's LLM-driven expansion steps.

Adapted from the crescendo port. The optimizer runs its vendored search steps
(decompose, mutate, feedback-rewrite, judge) on a worker thread through
:class:`~attack_anything_optimizer._llm.VendorLLMBridge`. This helper separates the
three failure classes that would otherwise be conflated:

* ``_BudgetSignal`` / ``_NoLLMSignal`` (the transport's budget/no-LLM markers) are
  re-raised untouched -- retrying a cap escape hides a truncated task, and the
  no-LLM case must fall through to the vendored rule-based paths.
* A transient transport failure is retried with exponential backoff and jitter.
* Anything that survives every attempt raises :class:`HelperLLMUnavailableError`
  so the caller decides, per call site, whether that is fatal.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from attack_anything_optimizer._llm import _BudgetSignal, _NoLLMSignal

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Total attempts (one initial call plus retries).
DEFAULT_ATTEMPTS = 3


class HelperLLMUnavailableError(RuntimeError):
    """An LLM-driven expansion step failed on every attempt."""


async def call_with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    stage: str,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = 1.0,
) -> T:
    """Call ``fn`` up to ``attempts`` times, then surface the failure.

    Raises:
        _BudgetSignal / _NoLLMSignal: propagated immediately, never retried.
        HelperLLMUnavailableError: every attempt failed.
    """
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except (_BudgetSignal, _NoLLMSignal):
            raise
        except Exception as exc:
            last = exc
            logger.warning(
                "AttackAnything: %s failed (attempt %d/%d)",
                stage,
                attempt,
                attempts,
                exc_info=True,
            )
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1)) * (0.5 + random.random())
            await asyncio.sleep(delay)

    assert last is not None
    raise HelperLLMUnavailableError(
        f"AttackAnything {stage} failed on all {attempts} attempts"
    ) from last


__all__ = ["HelperLLMUnavailableError", "call_with_retries", "DEFAULT_ATTEMPTS"]
