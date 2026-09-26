"""Tests for the bounded transient-failure retry."""

from __future__ import annotations

import pytest
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
    UnsupportedParamsError,
)

from superred.core.types.llm import BudgetExhaustedError, LLMUsage

from tap_optimizer import retry
from tap_optimizer.retry import retry_transient


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry, "_BASE_DELAY_SECONDS", 0.0)


def _calls(effects: list) -> tuple[list[int], object]:
    """A counting coroutine factory driven by a script of effects."""
    count = [0]

    async def call() -> str:
        count[0] += 1
        effect = effects[min(count[0] - 1, len(effects) - 1)]
        if isinstance(effect, BaseException):
            raise effect
        return effect

    return count, call


@pytest.mark.asyncio
async def test_returns_immediately_on_success() -> None:
    count, call = _calls(["ok"])

    assert await retry_transient(call, stage="s") == "ok"
    assert count[0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        APIConnectionError(message="x", llm_provider="bedrock", model="m"),
        InternalServerError(message="x", llm_provider="bedrock", model="m"),
        RateLimitError(message="x", llm_provider="bedrock", model="m"),
    ],
)
async def test_transient_errors_are_retried(error: Exception) -> None:
    count, call = _calls([error, error, "ok"])

    assert await retry_transient(call, stage="s") == "ok"
    assert count[0] == 3


@pytest.mark.asyncio
async def test_transient_error_is_raised_once_the_retries_are_spent() -> None:
    error = APIConnectionError(message="x", llm_provider="bedrock", model="m")
    count, call = _calls([error])

    with pytest.raises(APIConnectionError):
        await retry_transient(call, stage="s")
    assert count[0] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        # Verbatim provider message. TAP on-topic checks that died on this
        # used to be recorded as "on topic".
        UnsupportedParamsError(
            status_code=400,
            message="gpt-5 models don't support temperature=0.0",
        ),
        AuthenticationError(message="bad key", llm_provider="bedrock", model="m"),
        ValueError("unparseable"),
    ],
)
async def test_permanent_errors_are_raised_on_the_first_attempt(
    error: Exception,
) -> None:
    count, call = _calls([error])

    with pytest.raises(type(error)):
        await retry_transient(call, stage="s")
    assert count[0] == 1


@pytest.mark.asyncio
async def test_budget_exhaustion_is_never_retried() -> None:
    # Retrying the cost cap would spend past the cap the controller set.
    count, call = _calls([BudgetExhaustedError("spent", LLMUsage())])

    with pytest.raises(BudgetExhaustedError):
        await retry_transient(call, stage="s")
    assert count[0] == 1
