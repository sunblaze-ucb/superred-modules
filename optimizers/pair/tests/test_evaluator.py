"""Tests for PAIR's official judge behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from superred.core.types.llm import BudgetExhaustedError, LLMUsage

from pair_optimizer.evaluator import GCG_REFUSAL_KEYWORDS, PairEvaluator
from tests.conftest import mock_response


@pytest.mark.asyncio
async def test_gcg_judge_matches_official_binary_scoring() -> None:
    evaluator = PairEvaluator(llm=AsyncMock(), judge_method="gcg")
    refused = await evaluator.score_response(
        goal="goal",
        target_str="target",
        prompt="this prompt has enough words for the length gate",
        response="I'm sorry, I cannot help with that request at all",
    )
    accepted = await evaluator.score_response(
        goal="goal",
        target_str="target",
        prompt="this prompt has enough words for the length gate",
        response="Sure here are many detailed words in an answer",
    )
    assert refused.score == 1
    assert accepted.score == 10
    assert GCG_REFUSAL_KEYWORDS


@pytest.mark.asyncio
async def test_gpt_judge_uses_official_prompt_and_generation_defaults() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("Rating: [[7]]")
    evaluator = PairEvaluator(llm=llm, judge_method="gpt")

    result = await evaluator.score_response(
        goal="goal",
        target_str="target",
        prompt="attack prompt",
        response="assistant response",
    )

    assert result.score == 7
    assert result.normalized_score == pytest.approx(6 / 9)
    args, kwargs = llm.complete.call_args
    messages = args[0]
    assert messages[0]["role"] == "system"
    assert "Rate the response on a scale from 1 to 10" in messages[0]["content"]
    assert messages[1]["content"] == "[PROMPT]:attack prompt\n[ASSISTANT'S RESPONSE]: assistant response"
    assert kwargs == {"max_tokens": 10, "temperature": 0.0, "top_p": 1.0}


@pytest.mark.asyncio
async def test_gpt_judge_parse_failure_returns_one() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("not parseable")
    evaluator = PairEvaluator(llm=llm, judge_method="gpt")

    result = await evaluator.score_response(
        goal="goal",
        target_str="target",
        prompt="attack prompt",
        response="assistant response",
    )

    assert result.score == 1
    assert result.success is False


@pytest.mark.asyncio
async def test_budget_errors_propagate_from_gpt_judge() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = BudgetExhaustedError("budget", usage=LLMUsage(calls=1, cost=1.0))
    evaluator = PairEvaluator(llm=llm, judge_method="gpt")

    with pytest.raises(BudgetExhaustedError):
        await evaluator.score_response(
            goal="goal",
            target_str="target",
            prompt="attack prompt",
            response="assistant response",
        )
