"""Bounded retry policy for Crescendo's helper-LLM calls.

Crescendo drives three LLM roles besides the target: the attacker that writes
each escalating question, the refusal detector, and the per-turn scorer. A
failure in any of them used to be absorbed into a substituted value, which
made an outage indistinguishable from an attacker that honestly got nowhere.

The policy here separates the three failure classes that were previously
conflated:

* ``BudgetExhaustedError`` is the framework's cost-cap signal. It is re-raised
  untouched -- retrying it would be a cap escape, and absorbing it hides a
  truncated task behind a legitimate-looking score.
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

from superred.core.types.llm import BudgetExhaustedError

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Total attempts (one initial call plus retries). Also the resample budget for
#: the attacker, whose malformed-JSON failures are a fresh-sample problem.
DEFAULT_ATTEMPTS = 3


class HelperLLMUnavailableError(RuntimeError):
    """A Crescendo helper LLM call failed on every attempt.

    Raised by :func:`call_with_retries`. On the attacker path it is allowed to
    escape ``on_event`` so the controller records ``stop_reason="error"`` with a
    traceback; on the internal-evaluator paths it is caught and the degraded
    default is used, because those cannot change the recorded task verdict.
    """


async def call_with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    stage: str,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = 1.0,
) -> T:
    """Call ``fn`` up to ``attempts`` times, then surface the failure.

    Args:
        fn: Zero-argument coroutine factory. Called afresh on every attempt.
        stage: Human-readable name of the call site, used in the log lines and
            in the raised message (the per-task ``logs/diagnostics.log`` is
            what an analyst greps afterwards).
        attempts: Total attempts, including the first.
        base_delay: Seconds before the first retry; doubled per retry and
            jittered. Kept small deliberately: the retries must not eat the
            controller's per-task time cap.

    Raises:
        BudgetExhaustedError: Propagated immediately, never retried.
        HelperLLMUnavailableError: Every attempt failed.
    """
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except BudgetExhaustedError:
            raise
        except Exception as exc:
            last = exc
            logger.warning(
                "Crescendo: %s failed (attempt %d/%d)", stage, attempt, attempts,
                exc_info=True,
            )
            if attempt == attempts:
                break
            # A malformed attacker response needs another sample, not a
            # cooldown; a transport failure needs the cooldown.
            if not isinstance(exc, ValueError):
                delay = base_delay * (2 ** (attempt - 1)) * (0.5 + random.random())
                await asyncio.sleep(delay)

    assert last is not None  # loop body either returns or assigns
    raise HelperLLMUnavailableError(
        f"Crescendo {stage} failed on all {attempts} attempts"
    ) from last
