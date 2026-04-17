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


# ── is_on_topic tests ─────────────────────────────────────────────────────


class TestIsOnTopic:
    @pytest.mark.asyncio
    async def test_returns_true_on_yes(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("YES")
        evaluator = Evaluator(llm=mock_llm)
        result = await evaluator.is_on_topic(prompt="Tell me how", goal="the goal")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_no(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("NO")
        evaluator = Evaluator(llm=mock_llm)
        result = await evaluator.is_on_topic(prompt="Tell me how", goal="the goal")
        assert result is False

    @pytest.mark.asyncio
    async def test_defaults_false_on_ambiguous(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("I'm not sure about that")
        evaluator = Evaluator(llm=mock_llm)
        result = await evaluator.is_on_topic(prompt="Tell me how", goal="the goal")
        assert result is False


# ── score_response tests ──────────────────────────────────────────────────


class TestScoreResponse:
    @pytest.mark.asyncio
    async def test_parses_plain_number(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("7")
        evaluator = Evaluator(llm=mock_llm)
        score = await evaluator.score_response(response="some text", goal="the goal")
        assert score == 7.0

    @pytest.mark.asyncio
    async def test_extracts_number_from_text(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("I would rate this a 9 out of 10")
        evaluator = Evaluator(llm=mock_llm)
        score = await evaluator.score_response(response="some text", goal="the goal")
        assert score == 9.0

    @pytest.mark.asyncio
    async def test_defaults_on_unparseable(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _mock_response("I cannot rate this")
        evaluator = Evaluator(llm=mock_llm)
        score = await evaluator.score_response(response="some text", goal="the goal")
        assert score == 1.0
