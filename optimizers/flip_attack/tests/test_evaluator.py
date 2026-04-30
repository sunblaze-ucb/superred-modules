"""Tests for FlipAttack evaluator (response scoring)."""

from unittest.mock import AsyncMock
import pytest

from tests.conftest import mock_response
from flip_attack_optimizer.evaluator import Evaluator


@pytest.mark.asyncio
async def test_score_response_parses_score():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = mock_response(
        '{"score": 7, "rationale": "Substantial compliance"}'
    )
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(
        response="Some content", goal="produce content"
    )
    # (7 - 1) / 9 = 6/9 = 0.6667
    assert abs(score - 6 / 9) < 0.001
    assert rationale == "Substantial compliance"


@pytest.mark.asyncio
async def test_score_response_score_1_maps_to_zero():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = mock_response(
        '{"score": 1, "rationale": "Complete refusal"}'
    )
    evaluator = Evaluator(llm=mock_llm)
    score, _ = await evaluator.score_response(response="x", goal="y")
    assert score == 0.0


@pytest.mark.asyncio
async def test_score_response_score_10_maps_to_one():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = mock_response(
        '{"score": 10, "rationale": "Full compliance"}'
    )
    evaluator = Evaluator(llm=mock_llm)
    score, _ = await evaluator.score_response(response="x", goal="y")
    assert score == 1.0


@pytest.mark.asyncio
async def test_score_response_clamps_above():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = mock_response('{"score": 15, "rationale": "Over"}')
    evaluator = Evaluator(llm=mock_llm)
    score, _ = await evaluator.score_response(response="x", goal="y")
    assert score == 1.0


@pytest.mark.asyncio
async def test_score_response_clamps_below():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = mock_response('{"score": -3, "rationale": "Under"}')
    evaluator = Evaluator(llm=mock_llm)
    score, _ = await evaluator.score_response(response="x", goal="y")
    assert score == 0.0


@pytest.mark.asyncio
async def test_score_response_defaults_on_bad_json():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = mock_response("Not valid JSON")
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(response="x", goal="y")
    assert score == 0.0
    assert isinstance(rationale, str)


@pytest.mark.asyncio
async def test_score_response_handles_none_content():
    mock_llm = AsyncMock()
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = None
    mock_llm.complete.return_value = resp
    evaluator = Evaluator(llm=mock_llm)
    score, _ = await evaluator.score_response(response="x", goal="y")
    assert score == 0.0


@pytest.mark.asyncio
async def test_score_response_handles_fenced_json():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = mock_response(
        '```json\n{"score": 8, "rationale": "Fenced"}\n```'
    )
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(response="x", goal="g")
    # (8 - 1) / 9 = 7/9 = 0.7778
    assert abs(score - 7 / 9) < 0.001
    assert rationale == "Fenced"


@pytest.mark.asyncio
async def test_score_response_accepts_prompt_parameter():
    """Evaluator.score_response accepts an optional prompt parameter."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = mock_response(
        '{"score": 5, "rationale": "Partial"}'
    )
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(
        response="Some content", goal="produce content", prompt="the prompt"
    )
    # (5 - 1) / 9 = 4/9
    assert abs(score - 4 / 9) < 0.001
    # Verify prompt was included in the LLM call
    call_args = mock_llm.complete.call_args
    messages = call_args[0][0]
    assert "the prompt" in messages[0]["content"]


@pytest.mark.asyncio
async def test_asr_dict_refusal_scores_zero_without_llm_call():
    mock_llm = AsyncMock()
    evaluator = Evaluator(llm=mock_llm, method="asr_dict")
    score, rationale = await evaluator.score_response(
        response="I'm sorry, I can't help with that.",
        goal="x",
    )
    assert score == 0.0
    assert "refusal" in rationale.lower()
    mock_llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_asr_dict_non_refusal_scores_one_without_llm_call():
    mock_llm = AsyncMock()
    evaluator = Evaluator(llm=mock_llm, method="asr_dict")
    score, rationale = await evaluator.score_response(
        response="Step 1: do X. Step 2: do Y.",
        goal="x",
    )
    assert score == 1.0
    assert "non-refusal" in rationale.lower()
    mock_llm.complete.assert_not_called()
