"""Tests for Many-Shot evaluator."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from many_shot_optimizer.evaluator import Evaluator


def _mock_response(content):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


@pytest.mark.asyncio
async def test_score_response_parses_json():
    llm = AsyncMock()
    llm.complete.return_value = _mock_response('{"score": 0.7, "rationale": "Good"}')
    ev = Evaluator(llm=llm)
    score, rationale = await ev.score_response(response="answer", goal="goal")
    assert score == 0.7
    assert rationale == "Good"


@pytest.mark.asyncio
async def test_score_response_clamps_to_range():
    llm = AsyncMock()
    llm.complete.return_value = _mock_response('{"score": 1.5, "rationale": "Over"}')
    ev = Evaluator(llm=llm)
    score, _ = await ev.score_response(response="x", goal="g")
    assert score == 1.0


@pytest.mark.asyncio
async def test_score_response_handles_bad_json():
    llm = AsyncMock()
    llm.complete.return_value = _mock_response("not json at all")
    ev = Evaluator(llm=llm)
    score, rationale = await ev.score_response(response="x", goal="g")
    assert score == 0.0
    assert "Unparseable" in rationale


@pytest.mark.asyncio
async def test_score_response_handles_none_content():
    llm = AsyncMock()
    llm.complete.return_value = _mock_response(None)
    ev = Evaluator(llm=llm)
    score, _ = await ev.score_response(response="x", goal="g")
    assert score == 0.0


@pytest.mark.asyncio
async def test_score_response_handles_fenced_json():
    llm = AsyncMock()
    llm.complete.return_value = _mock_response(
        '```json\n{"score": 0.8, "rationale": "Fenced"}\n```'
    )
    ev = Evaluator(llm=llm)
    score, rationale = await ev.score_response(response="x", goal="g")
    assert score == 0.8
    assert rationale == "Fenced"
