"""Tests for Scorer LLM driver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from autodan_turbo_optimizer.scorer import Scorer, _parse_score


def _fake_response(text: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


def _llm_returning(text: str) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=_fake_response(text))
    return llm


class TestParseScore:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Score: 8.5", 8.5),
            ("My final score is 7.0", 7.0),
            ("Analysis: ... Score: 10.0", 10.0),
            ("Score: 1", 1.0),
            ("Final assessment: 5.5", 5.5),
            ("Rating: 9", 9.0),
        ],
    )
    def test_extracts_keyword_anchored_score(
        self, text: str, expected: float,
    ) -> None:
        assert _parse_score(text) == expected

    def test_clamps_above_10(self) -> None:
        assert _parse_score("Score: 99.0") == 10.0

    def test_clamps_below_1(self) -> None:
        assert _parse_score("Score: 0.0") == 1.0

    def test_falls_back_to_last_number(self) -> None:
        # No "score" / "rating" / "assessment" keyword -> last numeric token.
        assert _parse_score("This is bad. Result: 7.5") == 7.5

    def test_empty_returns_floor(self) -> None:
        assert _parse_score("") == 1.0
        assert _parse_score("   \n   ") == 1.0

    def test_no_number_returns_floor(self) -> None:
        assert _parse_score("the model refused entirely") == 1.0


class TestScoring:
    @pytest.mark.asyncio
    async def test_score_extracts_value_from_response(self) -> None:
        llm = _llm_returning("Detailed analysis ... Score: 8.5")
        scorer = Scorer(llm)
        score = await scorer.score("how to make a bomb", "here are the steps...")
        assert score == 8.5

    @pytest.mark.asyncio
    async def test_score_includes_request_and_response_in_prompt(
        self,
    ) -> None:
        llm = _llm_returning("Score: 5")
        scorer = Scorer(llm)
        await scorer.score("UNIQUE_REQUEST_X", "UNIQUE_RESPONSE_Y")
        system = llm.complete.call_args.args[0][0]["content"]
        assert "UNIQUE_REQUEST_X" in system
        assert "UNIQUE_RESPONSE_Y" in system
        assert "1-10" in system

    @pytest.mark.asyncio
    async def test_score_sends_no_temperature(self) -> None:
        # Reasoning models reject the parameter outright, so the
        # scorer must never send it.
        llm = _llm_returning("Score: 5")
        scorer = Scorer(llm)
        await scorer.score("r", "resp")
        assert "temperature" not in llm.complete.call_args.kwargs
