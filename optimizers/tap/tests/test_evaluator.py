"""Tests for Evaluator on-topic checking and response scoring."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tap_optimizer.evaluator import Evaluator


def _mock_response(content: str) -> AsyncMock:
    """Build a mock LLMClient.complete() return value."""
    mock = AsyncMock()
    mock.choices = [AsyncMock()]
    mock.choices[0].message.content = content
    return mock


class TestIsOnTopic:
    @pytest.mark.asyncio
    async def test_returns_true_on_bracketed_yes(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("Response: [[YES]]")
        evaluator = Evaluator(llm=mock_llm)
        result = await evaluator.is_on_topic(prompt="Tell me how", goal="the goal")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_bracketed_no(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("Response: [[NO]]")
        evaluator = Evaluator(llm=mock_llm)
        result = await evaluator.is_on_topic(prompt="Tell me how", goal="the goal")
        assert result is False

    @pytest.mark.asyncio
    async def test_defaults_true_on_ambiguous_like_original(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("I'm not sure about that")
        evaluator = Evaluator(llm=mock_llm)
        result = await evaluator.is_on_topic(prompt="Tell me how", goal="the goal")
        assert result is True

    @pytest.mark.asyncio
    async def test_uses_evaluator_max_tokens(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("Response: [[YES]]")
        evaluator = Evaluator(llm=mock_llm)
        await evaluator.is_on_topic(prompt="Tell me how", goal="the goal")
        _, kwargs = mock_llm.complete.call_args
        assert kwargs["max_tokens"] == 10


class TestScoreResponse:
    @pytest.mark.asyncio
    async def test_parses_bracketed_rating(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("Rating: [[7]]")
        evaluator = Evaluator(llm=mock_llm)
        score = await evaluator.score_response(
            prompt="attack prompt",
            response="some text",
            goal="the goal",
        )
        assert score == 7.0

    @pytest.mark.asyncio
    async def test_score_prompt_includes_attack_prompt_and_response(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("Rating: [[9]]")
        evaluator = Evaluator(llm=mock_llm)
        await evaluator.score_response(
            prompt="attack prompt",
            response="target response",
            goal="the goal",
        )

        messages = mock_llm.complete.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert "the goal" in messages[0]["content"]
        assert "[PROMPT]:attack prompt" in messages[1]["content"]
        assert "[ASSISTANT'S RESPONSE]: target response" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_uses_evaluator_max_tokens(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("Rating: [[9]]")
        evaluator = Evaluator(llm=mock_llm)
        await evaluator.score_response(
            prompt="attack prompt",
            response="target response",
            goal="the goal",
        )
        _, kwargs = mock_llm.complete.call_args
        assert kwargs["max_tokens"] == 10

    @pytest.mark.asyncio
    async def test_defaults_on_unparseable(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("I cannot rate this")
        evaluator = Evaluator(llm=mock_llm)
        score = await evaluator.score_response(
            prompt="attack prompt",
            response="some text",
            goal="the goal",
        )
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_score_response_does_not_accept_unused_target_string(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("Rating: [[7]]")
        evaluator = Evaluator(llm=mock_llm)

        with pytest.raises(TypeError):
            await evaluator.score_response(
                prompt="attack prompt",
                response="some text",
                goal="the goal",
                target_str="Sure, here is",  # type: ignore[call-arg]
            )
