"""Judge.judge() integration tests with a mocked litellm.

These cover the actual call site that other tests skipped: the
``judge`` async method that formats the prompt, calls
``litellm.acompletion``, and parses the verdict. The judge is exercised
directly (not via Task.evaluate) so failures point at the judge code,
not the task wiring.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from harmbench_claim.judge import HarmBenchJudge
from harmbench_claim.judge_failure import (
    JudgeUnavailableError,
    JudgeUnparseableError,
)


class UnsupportedParamsError(Exception):
    """Stands in for ``litellm.UnsupportedParamsError`` (classified by name)."""


class APIConnectionError(Exception):
    """Stands in for ``litellm.APIConnectionError``.

    The proxy used for RQ1.3-1 wrapped content-filter blocks, 5xx,
    timeouts and rate limits all in this one class.
    """


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record judge back-off delays instead of waiting for them."""
    delays: list[float] = []

    async def _record(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("harmbench_claim.judge_failure._default_sleep", _record)
    return delays


def _make_response(text: str) -> SimpleNamespace:
    """Build a minimal litellm-shaped response object."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


def _judge() -> HarmBenchJudge:
    return HarmBenchJudge(
        model="openai/test", api_base="https://x", api_key="sk-test",
    )


@pytest.mark.asyncio
async def test_judge_yes_verdict() -> None:
    """A 'Yes' from the underlying LLM produces is_harmful=True."""
    judge = _judge()
    with patch(
        "harmbench_claim.judge.acompletion",
        new=AsyncMock(return_value=_make_response("Yes")),
    ):
        verdict, raw = await judge.judge(
            behavior="some behavior", generation="model said yes-things",
        )
    assert verdict is True
    assert raw == "Yes"


@pytest.mark.asyncio
async def test_judge_no_verdict() -> None:
    judge = _judge()
    with patch(
        "harmbench_claim.judge.acompletion",
        new=AsyncMock(return_value=_make_response("No")),
    ):
        verdict, raw = await judge.judge(
            behavior="some behavior", generation="I'm sorry I cannot",
        )
    assert verdict is False
    assert raw == "No"


@pytest.mark.asyncio
async def test_judge_retries_an_unparseable_answer_then_raises(
    no_sleep: list[float],
) -> None:
    """"Maybe." is not a verdict, and must never become one.

    Judge output is non-deterministic (no temperature is pinned), so an
    occasional ramble is expected; the first retry is what stops that
    from costing a task. A second ramble is surfaced.
    """
    judge = HarmBenchJudge(
        model="openai/test", api_base="https://x", api_key="sk-test", max_attempts=2,
    )
    mock = AsyncMock(return_value=_make_response("Maybe."))
    with patch("harmbench_claim.judge.acompletion", new=mock):
        with pytest.raises(JudgeUnparseableError, match="Maybe."):
            await judge.judge(behavior="b", generation="g")
    assert mock.await_count == 2
    assert no_sleep == []  # rambling is not a rate problem


@pytest.mark.asyncio
async def test_judge_accepts_a_verdict_after_a_ramble() -> None:
    judge = HarmBenchJudge(
        model="openai/test", api_base="https://x", api_key="sk-test", max_attempts=2,
    )
    mock = AsyncMock(side_effect=[_make_response("Maybe."), _make_response("Yes")])
    with patch("harmbench_claim.judge.acompletion", new=mock):
        verdict, raw = await judge.judge(behavior="b", generation="g")
    assert verdict is True
    assert raw == "Yes"


@pytest.mark.asyncio
async def test_judge_does_not_retry_a_rejected_parameter(
    no_sleep: list[float],
) -> None:
    """25,010 HarmBench tasks died on this in the first sweep.

    HarmBench was the only one of the three judges that surfaced it
    rather than scoring 0, which is why its ASR was reported as None
    ("we measured nothing") instead of a confident fiction. Keep that
    behaviour, but stop paying for retries that cannot help.
    """
    judge = HarmBenchJudge(
        model="openai/test", api_base="https://x", api_key="sk-test", max_attempts=3,
    )
    mock = AsyncMock(
        side_effect=UnsupportedParamsError(
            "litellm.UnsupportedParamsError: gpt-5 models don't support "
            "temperature=0.0. Only temperature=1 is supported."
        )
    )
    with patch("harmbench_claim.judge.acompletion", new=mock):
        with pytest.raises(JudgeUnavailableError) as excinfo:
            await judge.judge(behavior="b", generation="g")
    assert mock.await_count == 1
    assert no_sleep == []
    assert excinfo.value.kind == "terminal"


@pytest.mark.asyncio
async def test_judge_does_not_retry_a_content_filter_block(
    no_sleep: list[float],
) -> None:
    """The provider refusing to grade the answer is not a verdict of "no"."""
    judge = _judge()
    mock = AsyncMock(
        side_effect=APIConnectionError(
            "litellm.APIConnectionError: Bedrock_mantleException - "
            '{"error":{"code":"validation_error","message":"Invalid prompt: '
            "we've limited access to this content for safety reasons.\"}}"
        )
    )
    with patch("harmbench_claim.judge.acompletion", new=mock):
        with pytest.raises(JudgeUnavailableError) as excinfo:
            await judge.judge(behavior="b", generation="g")
    assert mock.await_count == 1
    assert no_sleep == []
    assert excinfo.value.blocked_by_content_filter is True


@pytest.mark.asyncio
async def test_judge_retries_a_transient_failure(no_sleep: list[float]) -> None:
    judge = HarmBenchJudge(
        model="openai/test", api_base="https://x", api_key="sk-test", max_attempts=3,
    )
    mock = AsyncMock(
        side_effect=[
            APIConnectionError("Bedrock_mantleException - internal_server_error"),
            _make_response("Yes"),
        ]
    )
    with patch("harmbench_claim.judge.acompletion", new=mock):
        verdict, _ = await judge.judge(behavior="b", generation="g")
    assert verdict is True
    assert mock.await_count == 2
    assert len(no_sleep) == 1


@pytest.mark.asyncio
async def test_judge_passes_correct_kwargs_to_litellm() -> None:
    """Verify model, messages, max_tokens, api_base, api_key are all passed
    through correctly, and that NO temperature is sent (reasoning models such as
    gpt-5.x reject it, which would make the judge unusable)."""
    judge = _judge()
    mock_acompletion = AsyncMock(return_value=_make_response("Yes"))
    with patch("harmbench_claim.judge.acompletion", new=mock_acompletion):
        await judge.judge(behavior="BEHAVIOR_TEXT", generation="GEN_TEXT")

    call_kwargs = mock_acompletion.await_args.kwargs
    assert call_kwargs["model"] == "openai/test"
    assert "temperature" not in call_kwargs
    assert call_kwargs["max_tokens"] == 16
    # Retry policy is ours, not litellm's: num_retries retries terminal
    # errors too and cannot report that it gave up.
    assert "num_retries" not in call_kwargs
    assert call_kwargs["api_base"] == "https://x"
    assert call_kwargs["api_key"] == "sk-test"
    # Check messages structure
    messages = call_kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    # Behavior and generation must appear in the formatted prompt
    prompt = messages[0]["content"]
    assert "BEHAVIOR_TEXT" in prompt
    assert "GEN_TEXT" in prompt


@pytest.mark.asyncio
async def test_judge_uses_standard_template_when_no_context() -> None:
    judge = _judge()
    mock_acompletion = AsyncMock(return_value=_make_response("Yes"))
    with patch("harmbench_claim.judge.acompletion", new=mock_acompletion):
        await judge.judge(behavior="b", generation="g", context=None)
    prompt = mock_acompletion.await_args.kwargs["messages"][0]["content"]
    assert "[CONTEXT]:" not in prompt   # standard template has no context field


@pytest.mark.asyncio
async def test_judge_uses_contextual_template_when_context_present() -> None:
    judge = _judge()
    mock_acompletion = AsyncMock(return_value=_make_response("Yes"))
    with patch("harmbench_claim.judge.acompletion", new=mock_acompletion):
        await judge.judge(
            behavior="b", generation="g", context="THE_CONTEXT_VALUE",
        )
    prompt = mock_acompletion.await_args.kwargs["messages"][0]["content"]
    assert "[CONTEXT]:" in prompt
    assert "THE_CONTEXT_VALUE" in prompt


@pytest.mark.asyncio
async def test_judge_omits_api_base_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """If api_base is None and env var is not set, we don't pass it."""
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    judge = HarmBenchJudge(model="openai/test")
    mock_acompletion = AsyncMock(return_value=_make_response("Yes"))
    with patch("harmbench_claim.judge.acompletion", new=mock_acompletion):
        await judge.judge(behavior="b", generation="g")
    call_kwargs = mock_acompletion.await_args.kwargs
    assert "api_base" not in call_kwargs
    assert "api_key" not in call_kwargs


@pytest.mark.asyncio
async def test_judge_uses_env_credentials_when_args_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_API_BASE", "https://env-base/")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-env")
    judge = HarmBenchJudge(model="openai/test")
    mock_acompletion = AsyncMock(return_value=_make_response("Yes"))
    with patch("harmbench_claim.judge.acompletion", new=mock_acompletion):
        await judge.judge(behavior="b", generation="g")
    call_kwargs = mock_acompletion.await_args.kwargs
    assert call_kwargs["api_base"] == "https://env-base/"
    assert call_kwargs["api_key"] == "sk-env"


@pytest.mark.asyncio
async def test_judge_handles_none_content_in_response() -> None:
    """Some providers may return content=None on safety filters. Treat
    as unparseable rather than crashing on the .strip() call."""
    judge = _judge()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
    )
    with patch(
        "harmbench_claim.judge.acompletion", new=AsyncMock(return_value=response),
    ):
        with pytest.raises(JudgeUnparseableError):
            await judge.judge(behavior="b", generation="g")


@pytest.mark.asyncio
async def test_judge_handles_empty_choices_list() -> None:
    """Some providers return choices=[] when their content-policy filter
    intercepts the request at the API layer. Default behavior would be
    IndexError; we name the cause instead, and it reaches the caller as a
    judge failure rather than as a verdict of "not elicited"."""
    judge = _judge()
    response = SimpleNamespace(choices=[])
    with patch(
        "harmbench_claim.judge.acompletion", new=AsyncMock(return_value=response),
    ):
        with pytest.raises(JudgeUnavailableError, match="no choices"):
            await judge.judge(behavior="b", generation="g")
