"""Tests for Many-Shot example generator."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from many_shot_optimizer.generator import ExampleGenerator


def _mock_response(content):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


@pytest.mark.asyncio
async def test_generate_returns_requested_count():
    llm = AsyncMock()
    llm.complete.return_value = _mock_response(
        '[{"user": "Q1", "assistant": "A1"}, '
        '{"user": "Q2", "assistant": "A2"}, '
        '{"user": "Q3", "assistant": "A3"}]'
    )
    gen = ExampleGenerator(llm=llm)
    examples = await gen.generate(goal="test goal", count=3)
    assert len(examples) == 3
    assert all("user" in e and "assistant" in e for e in examples)


@pytest.mark.asyncio
async def test_generate_retries_on_partial():
    llm = AsyncMock()
    llm.complete.side_effect = [
        _mock_response('[{"user": "Q1", "assistant": "A1"}, {"user": "Q2", "assistant": "A2"}]'),
        _mock_response('[{"user": "Q3", "assistant": "A3"}, {"user": "Q4", "assistant": "A4"}]'),
    ]
    gen = ExampleGenerator(llm=llm)
    examples = await gen.generate(goal="test goal", count=4)
    assert len(examples) == 4


@pytest.mark.asyncio
async def test_generate_handles_bad_json():
    llm = AsyncMock()
    llm.complete.side_effect = [
        _mock_response("not json"),
        _mock_response('[{"user": "Q1", "assistant": "A1"}]'),
    ]
    gen = ExampleGenerator(llm=llm)
    examples = await gen.generate(goal="test goal", count=1)
    assert len(examples) == 1


@pytest.mark.asyncio
async def test_generate_gives_up_after_max_retries():
    llm = AsyncMock()
    llm.complete.return_value = _mock_response("bad json forever")
    gen = ExampleGenerator(llm=llm)
    examples = await gen.generate(goal="test goal", count=5)
    assert len(examples) < 5


@pytest.mark.asyncio
async def test_generate_handles_fenced_json():
    llm = AsyncMock()
    llm.complete.return_value = _mock_response(
        '```json\n[{"user": "Q1", "assistant": "A1"}]\n```'
    )
    gen = ExampleGenerator(llm=llm)
    examples = await gen.generate(goal="test", count=1)
    assert len(examples) == 1
