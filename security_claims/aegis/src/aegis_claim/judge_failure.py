"""Judge-failure policy shared by the chatbot benchmark claims.

A judge exists to answer one question: did the target comply? When it
cannot answer -- the provider rejected the call, blocked the judge's own
prompt, or the model replied with something that is not a verdict --
there is no answer, and the only honest outcomes are "try again" or "say
so loudly". Returning the minimum score instead manufactures a
measurement: it is byte-identical to a real "the target refused"
verdict, so nothing downstream can tell the two apart, and the
experiment ends up reporting a confident number computed from nothing.

The policy implemented here:

* Transient failures (rate limits, 5xx, timeouts, dropped connections)
  are retried up to ``max_attempts`` times with jittered exponential
  back-off.
* Terminal failures (a rejected parameter, bad credentials, a prompt the
  provider's own safety filter blocked) fail on the first attempt.
  Retrying a deterministic error only buys sleep: the first RQ1.3-1
  sweep spent roughly 1,700 hours retrying one rejected parameter
  423,140 times, for zero information.
* Either way the outcome is a raised :class:`JudgeFailedError`, never a
  score. The superred controller turns that into ``stop_reason="error"``
  with the traceback stored on the task, which is (a) excluded from the
  attack-success-rate denominator rather than counted as a failed
  attack, and (b) not a "keep" status for ``plan_resume``, so re-running
  the experiment re-measures the task. A returned-but-flagged verdict
  would instead be kept forever.

How an analyst distinguishes a judge failure from a real verdict: a
judge failure has no ``EvaluationResult`` at all. It appears only as a
task with ``stop_reason="error"`` whose ``error`` traceback names
``JudgeUnavailableError`` or ``JudgeUnparseableError``. Every persisted
``EvaluationResult`` these claims produce is, by construction, the
output of a judge that answered.

THIS FILE IS DUPLICATED VERBATIM across the three chatbot benchmark
claims (``harmbench_claim``, ``sorry_bench_claim``,
``strongreject_claim``). Each is an independently installable
distribution, so there is no shared package to hold it, and the three
judges are required to behave identically.
``tests/test_judge_failure_in_sync.py`` in each package fails if the
copies drift -- edit all three together.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Literal, TypeVar

from superred.core.types.llm import BudgetExhaustedError

T = TypeVar("T")

SleepFn = Callable[[float], Awaitable[None]]

FailureKind = Literal["transient", "terminal", "content_filter"]

DEFAULT_MAX_ATTEMPTS: int = 3
"""Attempts per judge call, including the first.

Three is the point where the marginal transient failure stops being
worth the wait: the failures that dominated the first sweep were
deterministic (a rejected parameter, a blocked prompt) and are now
classified terminal, so extra attempts would sleep without ever
changing the outcome.
"""

_BACKOFF_BASE_SECONDS: float = 1.0
_BACKOFF_MAX_SECONDS: float = 8.0
_DETAIL_MAX_CHARS: int = 500

# Providers do not agree on how to signal "I refuse to look at this
# content", and the litellm proxy used for RQ1.3-1 flattens every
# upstream error into ``APIConnectionError`` regardless of cause, so the
# exception TYPE carries almost no information. These markers are matched
# against the lower-cased exception text. The first entry is the exact
# phrase Bedrock returned on 8,126 judge calls in the first sweep.
_CONTENT_FILTER_MARKERS: tuple[str, ...] = (
    "limited access to this content for safety reasons",
    "flagged as potentially violating our usage policy",
    "responsibleaipolicyviolation",
    "content_filter",
    "content filter",
    "content policy",
    "content management policy",
)

# Deterministic: the same call will fail the same way forever. Matched
# against the exception CLASS NAME so no litellm import is required and
# so provider-specific subclasses are covered.
_TERMINAL_EXCEPTION_NAMES: frozenset[str] = frozenset(
    {
        "UnsupportedParamsError",
        "BadRequestError",
        "InvalidRequestError",
        "UnprocessableEntityError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "ContextWindowExceededError",
        "ContentPolicyViolationError",
    }
)


class JudgeFailedError(Exception):
    """The judge did not produce a usable verdict.

    Deliberately NOT a subclass of ``RuntimeError``: the call sites this
    replaces caught ``RuntimeError`` and turned it into a score of zero,
    and a judge failure must not be absorbable by a handler that was
    written before this policy existed.
    """

    def __init__(self, message: str, *, judge_model: str, attempts: int) -> None:
        super().__init__(message)
        self.judge_model = judge_model
        self.attempts = attempts


class JudgeUnavailableError(JudgeFailedError):
    """The judge call never returned an answer."""

    def __init__(
        self,
        *,
        judge_model: str,
        attempts: int,
        kind: FailureKind,
        detail: str,
    ) -> None:
        super().__init__(
            f"judge {judge_model!r} produced no verdict after {attempts} "
            f"attempt(s) [{kind}]: {_truncate(detail)}",
            judge_model=judge_model,
            attempts=attempts,
        )
        self.kind = kind
        self.detail = detail

    @property
    def blocked_by_content_filter(self) -> bool:
        """True when the provider refused to grade the content itself.

        This is the failure mode that biases results directionally: the
        judge prompt embeds the target's answer, so it is likeliest to be
        blocked precisely when the attack succeeded. Callers reporting
        error counts should report this one separately.
        """
        return self.kind == "content_filter"


class JudgeUnparseableError(JudgeFailedError):
    """The judge answered, but the answer is not a verdict."""

    def __init__(self, *, judge_model: str, attempts: int, raw_output: str) -> None:
        super().__init__(
            f"judge {judge_model!r} answered but produced no parseable verdict "
            f"in {attempts} attempt(s); last answer: {_truncate(raw_output)!r}",
            judge_model=judge_model,
            attempts=attempts,
        )
        self.raw_output = raw_output


def classify_judge_error(exc: BaseException) -> FailureKind:
    """Decide whether ``exc`` is worth retrying.

    Order matters. Content-filter blocks are checked first because the
    proxy reports them as connection errors, which would otherwise look
    transient.

    Everything not named terminal is treated as transient, including
    errors this function does not recognise. That asymmetry is
    deliberate: every failure that made retrying expensive in the first
    sweep is named in :data:`_TERMINAL_EXCEPTION_NAMES`, three bounded
    attempts cost a few seconds, and mistaking a real blip for a terminal
    error costs a whole task.
    """
    text = str(exc).lower()
    if any(marker in text for marker in _CONTENT_FILTER_MARKERS):
        return "content_filter"
    name = type(exc).__name__
    if name == "ContentPolicyViolationError":
        return "content_filter"
    if name in _TERMINAL_EXCEPTION_NAMES:
        return "terminal"
    return "transient"


async def run_judge(
    *,
    call: Callable[[], Awaitable[str]],
    parse: Callable[[str], T | None],
    judge_model: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sleep: SleepFn | None = None,
) -> tuple[T, str]:
    """Call a judge until it answers, or raise.

    Args:
        call: Performs one judge call and returns its raw text answer.
        parse: Turns a raw answer into a verdict, or ``None`` if the
            answer is not a verdict. A parse failure retries the call --
            a judge that rambled once may not ramble twice -- but does
            not sleep, because rambling is not a rate problem.
        judge_model: Model id, recorded on the raised exception so the
            persisted traceback names the judge that failed.
        max_attempts: Total attempts including the first. Must be >= 1.
        sleep: Injectable for tests; defaults to :func:`asyncio.sleep`.

    Returns:
        ``(verdict, raw_answer)``.

    Raises:
        JudgeUnavailableError: The call failed terminally, was blocked,
            or failed transiently on every attempt.
        JudgeUnparseableError: Every answer received was unparseable.
        BudgetExhaustedError: Propagated untouched — the controller
            handles it as ``stop_reason="budget_exhausted"``, which is a
            deliberate stop, not a broken judge.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    sleep_fn = sleep if sleep is not None else _default_sleep

    last_detail = ""
    last_raw: str | None = None
    last_failure_was_parse = False
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            raw = await call()
        except BudgetExhaustedError:
            raise
        except Exception as exc:
            kind = classify_judge_error(exc)
            if kind != "transient":
                raise JudgeUnavailableError(
                    judge_model=judge_model,
                    attempts=attempt,
                    kind=kind,
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc
            last_detail = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            last_failure_was_parse = False
            last_exc = exc
            if attempt < max_attempts:
                await sleep_fn(_backoff_delay(attempt))
            continue

        verdict = parse(raw)
        if verdict is not None:
            return verdict, raw
        last_raw = raw
        last_detail = f"attempt {attempt}: unparseable answer ({len(raw)} chars)"
        last_failure_was_parse = True
        last_exc = None

    if last_failure_was_parse:
        raise JudgeUnparseableError(
            judge_model=judge_model,
            attempts=max_attempts,
            raw_output=last_raw if last_raw is not None else "",
        )
    error = JudgeUnavailableError(
        judge_model=judge_model,
        attempts=max_attempts,
        kind="transient",
        detail=last_detail,
    )
    raise error from last_exc


async def _default_sleep(seconds: float) -> None:
    """Indirection so tests can observe back-off without waiting for it."""
    await asyncio.sleep(seconds)


def _backoff_delay(attempt: int) -> float:
    """Exponential back-off with jitter, so parallel tasks do not resync."""
    base = min(_BACKOFF_BASE_SECONDS * 2.0 ** (attempt - 1), _BACKOFF_MAX_SECONDS)
    return base * (1.0 + random.random() * 0.25)


def _truncate(text: str) -> str:
    if len(text) <= _DETAIL_MAX_CHARS:
        return text
    return text[:_DETAIL_MAX_CHARS] + "..."


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "FailureKind",
    "JudgeFailedError",
    "JudgeUnavailableError",
    "JudgeUnparseableError",
    "classify_judge_error",
    "run_judge",
]
