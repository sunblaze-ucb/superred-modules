"""Tests for FlipAttack evaluator (response scoring)."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from flip_attack_optimizer.evaluator import Evaluator


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


@pytest.mark.asyncio
async def test_score_response_parses_score():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(
        '{"score": 0.7, "rationale": "Partial compliance"}'
    )
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(
        response="Some content", goal="produce content"
    )
    assert score == 0.7
    assert rationale == "Partial compliance"


@pytest.mark.asyncio
async def test_score_response_clamps_above():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response('{"score": 1.5, "rationale": "Over"}')
    evaluator = Evaluator(llm=mock_llm)
    score, _ = await evaluator.score_response(response="x", goal="y")
    assert score == 1.0


@pytest.mark.asyncio
async def test_score_response_clamps_below():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response('{"score": -0.3, "rationale": "Under"}')
    evaluator = Evaluator(llm=mock_llm)
    score, _ = await evaluator.score_response(response="x", goal="y")
    assert score == 0.0


@pytest.mark.asyncio
async def test_score_response_defaults_on_bad_json():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response("Not valid JSON")
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(response="x", goal="y")
    assert score == 0.0
    assert isinstance(rationale, str)


@pytest.mark.asyncio
async def test_score_response_handles_none_content():
    mock_llm = AsyncMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = None
    mock_llm.complete.return_value = resp
    evaluator = Evaluator(llm=mock_llm)
    score, _ = await evaluator.score_response(response="x", goal="y")
    assert score == 0.0


@pytest.mark.asyncio
async def test_score_response_handles_integer_score():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response('{"score": 1, "rationale": "Full"}')
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(response="x", goal="y")
    assert score == 1.0
    assert rationale == "Full"
