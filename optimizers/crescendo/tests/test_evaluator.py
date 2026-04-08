"""Tests for the Crescendo evaluator (refusal detection + scoring)."""

from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from crescendo_optimizer.evaluator import Evaluator


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


@pytest.mark.asyncio
async def test_is_refusal_detects_refusal():
    evaluator = Evaluator(model="gpt-4")
    with patch("crescendo_optimizer.evaluator.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response("YES")
        result = await evaluator.is_refusal("I can't help with that.")
        assert result is True


@pytest.mark.asyncio
async def test_is_refusal_detects_non_refusal():
    evaluator = Evaluator(model="gpt-4")
    with patch("crescendo_optimizer.evaluator.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response("NO")
        result = await evaluator.is_refusal("Here is the information you requested...")
        assert result is False


@pytest.mark.asyncio
async def test_is_refusal_defaults_false_on_ambiguous():
    evaluator = Evaluator(model="gpt-4")
    with patch("crescendo_optimizer.evaluator.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response("Maybe, it's unclear")
        result = await evaluator.is_refusal("Some response")
        assert result is False


@pytest.mark.asyncio
async def test_score_response_parses_score():
    evaluator = Evaluator(model="gpt-4")
    with patch("crescendo_optimizer.evaluator.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response(
            '{"score": 0.7, "rationale": "Partial compliance"}'
        )
        score, rationale = await evaluator.score_response(
            response="Some harmful content", goal="produce harmful content"
        )
        assert score == 0.7
        assert rationale == "Partial compliance"


@pytest.mark.asyncio
async def test_score_response_clamps_to_range():
    evaluator = Evaluator(model="gpt-4")
    with patch("crescendo_optimizer.evaluator.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response('{"score": 1.5, "rationale": "Over"}')
        score, _ = await evaluator.score_response(response="x", goal="y")
        assert score == 1.0

        mock.return_value = _mock_response('{"score": -0.3, "rationale": "Under"}')
        score, _ = await evaluator.score_response(response="x", goal="y")
        assert score == 0.0


@pytest.mark.asyncio
async def test_score_response_defaults_on_bad_json():
    evaluator = Evaluator(model="gpt-4")
    with patch("crescendo_optimizer.evaluator.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response("I think this scores about 0.5")
        score, rationale = await evaluator.score_response(response="x", goal="y")
        assert score == 0.0
        assert isinstance(rationale, str)


@pytest.mark.asyncio
async def test_score_response_handles_integer_score():
    evaluator = Evaluator(model="gpt-4")
    with patch("crescendo_optimizer.evaluator.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response('{"score": 1, "rationale": "Full match"}')
        score, rationale = await evaluator.score_response(response="x", goal="y")
        assert score == 1.0
        assert rationale == "Full match"
