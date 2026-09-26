"""The shared judge-failure policy.

Lives here rather than being copied into all three claim packages
(``test_judge_failure_in_sync.py`` proves the policy code is identical
everywhere, so testing it once tests it everywhere). The error strings
below are verbatim provider messages.
"""

from __future__ import annotations

import pytest
from superred.core.types.llm import BudgetExhaustedError, LLMUsage

from strongreject_claim.judge_failure import (
    JudgeUnavailableError,
    JudgeUnparseableError,
    classify_judge_error,
    run_judge,
)

# ---------------------------------------------------------------------------
# Stand-ins for litellm exception types. Classification is by class NAME,
# so these exercise the real branches without importing litellm.
# ---------------------------------------------------------------------------


class UnsupportedParamsError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class RateLimitError(Exception):
    pass


class ContentPolicyViolationError(Exception):
    pass


# Verbatim provider messages: a rejected parameter and a content-filter
# block of the judge's own prompt.
REJECTED_PARAMETER = (
    "litellm.UnsupportedParamsError: gpt-5 models don't support "
    "temperature=0.0. Only temperature=1 is supported."
)
CONTENT_FILTER_BLOCK = (
    "litellm.APIConnectionError: Bedrock_mantleException - "
    '{"error":{"code":"validation_error","message":"Invalid prompt: we\'ve '
    'limited access to this content for safety reasons."}}'
)
PROVIDER_5XX = (
    "litellm.APIConnectionError: Bedrock_mantleException - "
    '{"error":{"code":"internal_server_error"}}'
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_rejected_parameter_is_terminal() -> None:
    assert classify_judge_error(UnsupportedParamsError(REJECTED_PARAMETER)) == "terminal"


def test_content_filter_block_is_recognised_by_message_not_type() -> None:
    """The proxy reports a block as a connection error.

    Classifying on type alone would call this transient and retry a
    deterministic refusal; classifying on the message gets it right.
    """
    assert classify_judge_error(APIConnectionError(CONTENT_FILTER_BLOCK)) == "content_filter"


def test_provider_5xx_in_the_same_exception_class_is_transient() -> None:
    assert classify_judge_error(APIConnectionError(PROVIDER_5XX)) == "transient"


def test_content_policy_exception_type_is_a_block_even_without_a_message() -> None:
    assert classify_judge_error(ContentPolicyViolationError("")) == "content_filter"


def test_unrecognised_error_is_transient() -> None:
    """Unknown means "worth one more try", not "give up".

    Every known failure that makes retrying expensive is named terminal;
    the cost of retrying an unknown one is seconds, while the cost of
    abandoning a real blip is a whole task.
    """
    assert classify_judge_error(ValueError("something new")) == "transient"


# ---------------------------------------------------------------------------
# run_judge
# ---------------------------------------------------------------------------


async def _never_called() -> str:  # pragma: no cover - guard for bad wiring
    raise AssertionError("judge should not have been called")


async def test_returns_the_first_parseable_verdict() -> None:
    calls = 0

    async def _call() -> str:
        nonlocal calls
        calls += 1
        return "yes"

    verdict, raw = await run_judge(
        call=_call, parse=lambda t: t == "yes", judge_model="m", max_attempts=3
    )
    assert verdict is True
    assert raw == "yes"
    assert calls == 1


async def test_falsy_verdicts_are_verdicts() -> None:
    """``False`` and ``0.0`` are answers; only ``None`` means "no answer"."""
    verdict, _ = await run_judge(
        call=lambda: _returns("no"), parse=lambda t: t == "yes", judge_model="m"
    )
    assert verdict is False


async def test_terminal_failure_costs_one_call_and_no_sleep() -> None:
    calls = 0
    delays: list[float] = []

    async def _call() -> str:
        nonlocal calls
        calls += 1
        raise UnsupportedParamsError(REJECTED_PARAMETER)

    async def _sleep(seconds: float) -> None:
        delays.append(seconds)

    with pytest.raises(JudgeUnavailableError) as excinfo:
        await run_judge(
            call=_call,
            parse=lambda t: t,
            judge_model="openai/gpt-5.4",
            max_attempts=5,
            sleep=_sleep,
        )
    assert calls == 1
    assert delays == []
    assert excinfo.value.kind == "terminal"
    assert excinfo.value.judge_model == "openai/gpt-5.4"
    assert "openai/gpt-5.4" in str(excinfo.value)


async def test_transient_failure_backs_off_and_then_surfaces() -> None:
    calls = 0
    delays: list[float] = []

    async def _call() -> str:
        nonlocal calls
        calls += 1
        raise RateLimitError("too many tokens")

    async def _sleep(seconds: float) -> None:
        delays.append(seconds)

    with pytest.raises(JudgeUnavailableError) as excinfo:
        await run_judge(
            call=_call, parse=lambda t: t, judge_model="m", max_attempts=3, sleep=_sleep
        )
    assert calls == 3
    assert len(delays) == 2
    assert delays[1] > delays[0]  # exponential, not flat
    assert excinfo.value.kind == "transient"
    assert isinstance(excinfo.value.__cause__, RateLimitError)


async def test_transient_failure_that_recovers_returns_the_real_verdict() -> None:
    calls = 0

    async def _call() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitError("too many tokens")
        return "yes"

    async def _sleep(seconds: float) -> None:
        return None

    verdict, _ = await run_judge(
        call=_call, parse=lambda t: t == "yes", judge_model="m", sleep=_sleep
    )
    assert verdict is True
    assert calls == 2


async def test_unparseable_answer_buys_one_more_call_but_no_sleep() -> None:
    """A judge that rambled once may answer cleanly next time.

    Rambling is not a rate problem, so retrying it must not sleep.
    """
    calls = 0
    delays: list[float] = []

    async def _call() -> str:
        nonlocal calls
        calls += 1
        return "Maybe."

    async def _sleep(seconds: float) -> None:
        delays.append(seconds)

    with pytest.raises(JudgeUnparseableError) as excinfo:
        await run_judge(
            call=_call,
            parse=lambda t: None,
            judge_model="m",
            max_attempts=3,
            sleep=_sleep,
        )
    assert calls == 3
    assert delays == []
    assert excinfo.value.raw_output == "Maybe."


async def test_unparseable_then_parseable_returns_the_verdict() -> None:
    answers = iter(["Maybe.", "yes"])

    async def _call() -> str:
        return next(answers)

    verdict, _ = await run_judge(
        call=_call,
        parse=lambda t: True if t == "yes" else None,
        judge_model="m",
        max_attempts=3,
    )
    assert verdict is True


async def test_budget_exhaustion_propagates_untouched() -> None:
    """A spent budget is a deliberate stop, not a broken judge.

    The controller maps it to ``stop_reason="budget_exhausted"``;
    wrapping it would hide that behind a judge error.
    """

    async def _call() -> str:
        raise BudgetExhaustedError("budget spent", usage=LLMUsage())

    with pytest.raises(BudgetExhaustedError):
        await run_judge(call=_call, parse=lambda t: t, judge_model="m")


async def test_max_attempts_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        await run_judge(
            call=_never_called, parse=lambda t: t, judge_model="m", max_attempts=0
        )


async def _returns(value: str) -> str:
    return value
