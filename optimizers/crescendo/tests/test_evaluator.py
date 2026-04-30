"""Tests for the Crescendo evaluator (refusal detection + scoring)."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from crescendo_optimizer.evaluator import Evaluator


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


@pytest.mark.asyncio
async def test_is_refusal_detects_refusal():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response("YES")
    evaluator = Evaluator(llm=mock_llm)
    result = await evaluator.is_refusal("I can't help with that.")
    assert result is True


@pytest.mark.asyncio
async def test_is_refusal_detects_non_refusal():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response("NO")
    evaluator = Evaluator(llm=mock_llm)
    result = await evaluator.is_refusal("Here is the information you requested...")
    assert result is False


@pytest.mark.asyncio
async def test_is_refusal_defaults_false_on_ambiguous():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response("Maybe, it's unclear")
    evaluator = Evaluator(llm=mock_llm)
    result = await evaluator.is_refusal("Some response")
    assert result is False


@pytest.mark.asyncio
async def test_score_response_parses_score():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(
        '{"score_value": "70", "description": "Partial", "rationale": "Partial compliance"}'
    )
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(
        response="Some harmful content", goal="produce harmful content"
    )
    assert score == 0.7
    assert rationale == "Partial compliance"


@pytest.mark.asyncio
async def test_score_response_clamps_to_range():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(
        '{"score_value": "150", "description": "Over", "rationale": "Over"}'
    )
    evaluator = Evaluator(llm=mock_llm)
    score, _ = await evaluator.score_response(response="x", goal="y")
    assert score == 1.0

    mock_llm.complete.return_value = _mock_response(
        '{"score_value": "-30", "description": "Under", "rationale": "Under"}'
    )
    score, _ = await evaluator.score_response(response="x", goal="y")
    assert score == 0.0


@pytest.mark.asyncio
async def test_score_response_defaults_on_bad_json():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response("I think this scores about 50")
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(response="x", goal="y")
    assert score == 0.0
    assert isinstance(rationale, str)


@pytest.mark.asyncio
async def test_score_response_handles_integer_score_value():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(
        '{"score_value": "100", "description": "Full", "rationale": "Full match"}'
    )
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(response="x", goal="y")
    assert score == 1.0
    assert rationale == "Full match"


@pytest.mark.asyncio
async def test_score_response_handles_decimal_string_score_value():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(
        '{"score_value": "80.5", "description": "High", "rationale": "Decimal string"}'
    )
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(response="x", goal="y")
    assert score == 0.805
    assert rationale == "Decimal string"


@pytest.mark.asyncio
async def test_score_response_handles_fenced_json():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(
        '```json\n{"score_value": "80", "description": "High", "rationale": "Fenced"}\n```'
    )
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(response="x", goal="g")
    assert score == 0.8
    assert rationale == "Fenced"


@pytest.mark.asyncio
async def test_score_response_normalizes_zero():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(
        '{"score_value": "0", "description": "Refusal", "rationale": "Complete refusal"}'
    )
    evaluator = Evaluator(llm=mock_llm)
    score, rationale = await evaluator.score_response(response="x", goal="y")
    assert score == 0.0
    assert rationale == "Complete refusal"


@pytest.mark.asyncio
async def test_score_response_always_uses_goal_as_task():
    """Scoring should always evaluate against the adversarial goal."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(
        '{"score_value": "50", "description": "Mid", "rationale": "Mid-range"}'
    )
    evaluator = Evaluator(llm=mock_llm)
    await evaluator.score_response(
        response="Some response", goal="the adversarial goal"
    )
    messages = mock_llm.complete.call_args[0][0]
    user_msg = messages[-1]["content"]
    assert "the adversarial goal" in user_msg


@pytest.mark.asyncio
async def test_score_response_sends_system_and_user_messages():
    """Score request should use system + user message pattern."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(
        '{"score_value": "60", "description": "Moderate", "rationale": "Some progress"}'
    )
    evaluator = Evaluator(llm=mock_llm)
    await evaluator.score_response(response="x", goal="y")
    messages = mock_llm.complete.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
