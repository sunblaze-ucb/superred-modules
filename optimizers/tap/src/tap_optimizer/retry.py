"""Bounded retry for the attacker-side helper LLM calls.

Why this exists: every helper call TAP makes (prompt generation, the
on-topic check, the 1-10 judge) used to be wrapped in a bare
``except Exception`` that substituted a value: a pruned node, "on
topic", score 1.0. A provider hiccup was therefore recorded as a
search decision, and a permanent misconfiguration was recorded as a
thorough search that found nothing.

The three failure classes are separated here, once:

* the cost cap (``BudgetExhaustedError``) is a framework decision, so
  it is re-raised untouched; retrying it would spend past the cap,
* a transient provider failure is retried a bounded number of times,
* anything else, and anything that survives the retries, is raised so
  the caller can decide loudly instead of guessing quietly.

Deliberately duplicated in the ``autodan_turbo`` package: the two
optimizers ship as independent distributions and cannot import from
each other. The shared home is ``superred.core`` once the framework
grows one.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)
from superred.core.types.llm import BudgetExhaustedError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Provider-side failures a second attempt can plausibly fix. ``litellm.Timeout``
# subclasses ``APIConnectionError``, so it is covered. Everything else (a
# rejected parameter, a bad key, an unparseable answer) is permanent for the
# life of the task and must not be retried.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)

DEFAULT_RETRIES = 2

# Module-level so tests can shrink them; the caps keep the backoff far below the
# controller's per-task time cap, since a retried call may itself be a timeout.
_BASE_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 8.0


async def retry_transient(
    call: Callable[[], Awaitable[T]],
    *,
    stage: str,
    retries: int = DEFAULT_RETRIES,
) -> T:
    """Await ``call()``, retrying transient provider failures.

    Args:
        call: Idempotent zero-argument coroutine factory. It is invoked
            at most ``retries + 1`` times, so it must not mutate state
            that a second attempt would corrupt.
        stage: Short label for log lines, e.g. ``"attacker"``.
        retries: Extra attempts after the first.

    Returns:
        Whatever ``call()`` returns.

    Raises:
        BudgetExhaustedError: Immediately and untouched, so the
            controller records ``stop_reason="budget_exhausted"``.
        Exception: The final failure, once the retries are spent or the
            failure is not transient.
    """
    attempt = 0
    while True:
        try:
            return await call()
        except BudgetExhaustedError:
            raise
        except TRANSIENT_ERRORS as exc:
            if attempt >= retries:
                logger.warning(
                    "TAP: %s gave up after %d transient failures",
                    stage,
                    attempt + 1,
                    exc_info=True,
                )
                raise
            delay = min(_MAX_DELAY_SECONDS, _BASE_DELAY_SECONDS * (2**attempt))
            logger.info(
                "TAP: %s hit %s, retrying in %.1fs (attempt %d of %d)",
                stage,
                type(exc).__name__,
                delay,
                attempt + 1,
                retries + 1,
            )
            await asyncio.sleep(delay * random.uniform(0.5, 1.5))
            attempt += 1


__all__ = ["DEFAULT_RETRIES", "TRANSIENT_ERRORS", "retry_transient"]
