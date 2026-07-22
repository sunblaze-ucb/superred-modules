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
async def test_judge_unparseable_raises() -> None:
    """If the underlying LLM emits something we can't parse, we raise
    RuntimeError (rather than silently defaulting to False)."""
    judge = _judge()
    with patch(
        "harmbench_claim.judge.acompletion",
        new=AsyncMock(return_value=_make_response("Maybe.")),
    ):
        with pytest.raises(RuntimeError, match="unparseable judge output"):
            await judge.judge(behavior="b", generation="g")


@pytest.mark.asyncio
async def test_judge_passes_correct_kwargs_to_litellm() -> None:
    """Verify model, messages, temperature, max_tokens, api_base, api_key
    are all passed through correctly."""
    judge = _judge()
    mock_acompletion = AsyncMock(return_value=_make_response("Yes"))
    with patch("harmbench_claim.judge.acompletion", new=mock_acompletion):
        await judge.judge(behavior="BEHAVIOR_TEXT", generation="GEN_TEXT")

    call_kwargs = mock_acompletion.await_args.kwargs
    assert call_kwargs["model"] == "openai/test"
    assert call_kwargs["temperature"] == 0
    assert call_kwargs["max_tokens"] == 16
    assert call_kwargs["num_retries"] == 2
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
        with pytest.raises(RuntimeError):
            await judge.judge(behavior="b", generation="g")


@pytest.mark.asyncio
async def test_judge_handles_empty_choices_list() -> None:
    """Some providers return choices=[] when their content-policy filter
    intercepts the request at the API layer. Default behavior would be
    IndexError; we surface as RuntimeError so the task layer treats it
    as 'unparseable judge output' (label=0) with an informative message."""
    judge = _judge()
    response = SimpleNamespace(choices=[])
    with patch(
        "harmbench_claim.judge.acompletion", new=AsyncMock(return_value=response),
    ):
        with pytest.raises(RuntimeError, match="no choices"):
            await judge.judge(behavior="b", generation="g")
