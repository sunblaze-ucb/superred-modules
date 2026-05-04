"""Tests for Many-Shot evaluator."""

from unittest.mock import AsyncMock
import pytest

from tests.conftest import mock_response
from many_shot_optimizer.evaluator import Evaluator


@pytest.mark.asyncio
async def test_refusal_classifier_scores_compliance_as_success():
    llm = AsyncMock()
    llm.complete.return_value = mock_response(
        '{"refusal": false, "rationale": "The answer attempts the requested behavior."}'
    )
    ev = Evaluator(llm=llm)
    score, rationale = await ev.score_response(response="answer", goal="goal")
    assert score == 1.0
    assert "attempts" in rationale


@pytest.mark.asyncio
async def test_refusal_classifier_scores_refusal_as_failure():
    llm = AsyncMock()
    llm.complete.return_value = mock_response(
        '{"refusal": true, "rationale": "The answer refuses."}'
    )
    ev = Evaluator(llm=llm)
    score, rationale = await ev.score_response(response="answer", goal="goal")
    assert score == 0.0
    assert "refuses" in rationale


@pytest.mark.asyncio
async def test_refusal_classifier_accepts_boolean_like_strings():
    llm = AsyncMock()
    llm.complete.return_value = mock_response(
        '{"refusal": "false", "rationale": "The answer attempts the requested behavior."}'
    )
    ev = Evaluator(llm=llm)
    score, _ = await ev.score_response(response="answer", goal="goal")
    assert score == 1.0


@pytest.mark.asyncio
async def test_refusal_classifier_falls_back_to_refusal_strings():
    llm = AsyncMock()
    llm.complete.return_value = mock_response("not json")
    ev = Evaluator(llm=llm)
    score, rationale = await ev.score_response(
        response="I can't help with that request.", goal="goal"
    )
    assert score == 0.0
    assert "refusal" in rationale.lower()


@pytest.mark.asyncio
async def test_llm_judge_score_response_parses_json():
    llm = AsyncMock()
    llm.complete.return_value = mock_response('{"score": 0.7, "rationale": "Good"}')
    ev = Evaluator(llm=llm, method="llm_judge")
    score, rationale = await ev.score_response(response="answer", goal="goal")
    assert score == 0.7
    assert rationale == "Good"


@pytest.mark.asyncio
async def test_llm_judge_score_response_clamps_to_range():
    llm = AsyncMock()
    llm.complete.return_value = mock_response('{"score": 1.5, "rationale": "Over"}')
    ev = Evaluator(llm=llm, method="llm_judge")
    score, _ = await ev.score_response(response="x", goal="g")
    assert score == 1.0


@pytest.mark.asyncio
async def test_llm_judge_score_response_handles_bad_json():
    llm = AsyncMock()
    llm.complete.return_value = mock_response("not json at all")
    ev = Evaluator(llm=llm, method="llm_judge")
    score, rationale = await ev.score_response(response="x", goal="g")
    assert score == 0.0
    assert "Unparseable" in rationale


@pytest.mark.asyncio
async def test_llm_judge_score_response_handles_none_content():
    llm = AsyncMock()
    llm.complete.return_value = mock_response(None)
    ev = Evaluator(llm=llm, method="llm_judge")
    score, _ = await ev.score_response(response="x", goal="g")
    assert score == 0.0


@pytest.mark.asyncio
async def test_llm_judge_score_response_handles_fenced_json():
    llm = AsyncMock()
    llm.complete.return_value = mock_response(
        '```json\n{"score": 0.8, "rationale": "Fenced"}\n```'
    )
    ev = Evaluator(llm=llm, method="llm_judge")
    score, rationale = await ev.score_response(response="x", goal="g")
    assert score == 0.8
    assert rationale == "Fenced"
