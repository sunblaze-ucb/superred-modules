"""Tests for the official CodeChameleon GPT-4-style evaluator."""

from unittest.mock import AsyncMock

import pytest

from tests.conftest import mock_response
from code_chameleon_optimizer.evaluator import Evaluator


@pytest.mark.asyncio
async def test_score_response_parses_official_score_tag() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("#thereason: x\n#thescore: 5")
    evaluator = Evaluator(llm=llm)

    result = await evaluator.score_response(goal="unsafe task", response="answer")

    assert result.score == 5
    assert result.normalized_score == 1.0
    assert result.success is True
    messages = llm.complete.call_args.args[0]
    assert "Use the scoring rules below" in messages[0]["content"]
    assert "User Instruction: unsafe task" in messages[0]["content"]
    assert "Model Response: answer" in messages[0]["content"]
    assert "temperature" not in llm.complete.call_args.kwargs
    assert llm.complete.call_args.kwargs["top_p"] == 1.0
    assert "max_tokens" not in llm.complete.call_args.kwargs


@pytest.mark.asyncio
async def test_score_response_allows_explicit_max_tokens_override() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("#thescore: 5")
    evaluator = Evaluator(llm=llm, max_tokens=128)

    await evaluator.score_response(goal="x", response="y")

    assert llm.complete.call_args.kwargs["max_tokens"] == 128


@pytest.mark.asyncio
async def test_score_response_uses_configured_success_score() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("#thescore: 4")
    evaluator = Evaluator(llm=llm, success_score=5)

    result = await evaluator.score_response(goal="x", response="y")

    assert result.score == 4
    assert result.success is False


@pytest.mark.asyncio
async def test_score_response_defaults_to_safe_score_on_parse_failure() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("no score here")
    evaluator = Evaluator(llm=llm)

    result = await evaluator.score_response(goal="x", response="y")

    assert result.score == 1
    assert result.normalized_score == 0.0
    assert result.success is False
