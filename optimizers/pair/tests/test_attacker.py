"""Tests for PAIR attacker generation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from superred.core.types.llm import BudgetExhaustedError, LLMUsage

from pair_optimizer.attacker import PairAttacker, PairStream
from tests.conftest import mock_response


@pytest.mark.asyncio
async def test_attacker_uses_official_generation_defaults() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response('{"improvement": "better", "prompt": "attack"}')
    stream = PairStream(index=0, system_prompt="system", processed_response="init")
    attacker = PairAttacker(llm=llm)

    proposal = await attacker.generate(stream)

    assert proposal.prompt == "attack"
    args, kwargs = llm.complete.call_args
    assert args[0] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "init"},
    ]
    assert kwargs == {"max_tokens": 500, "temperature": 1.0, "top_p": 0.9, "stop": ["}"]}


@pytest.mark.asyncio
async def test_attacker_retries_invalid_json() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [
        mock_response("not json"),
        mock_response('{"improvement": "better", "prompt": "attack"}'),
    ]
    stream = PairStream(index=0, system_prompt="system", processed_response="init")
    attacker = PairAttacker(llm=llm)

    proposal = await attacker.generate(stream)

    assert proposal.prompt == "attack"
    assert llm.complete.call_count == 2


@pytest.mark.asyncio
async def test_attacker_missing_system_prompt_is_allowed() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response('{"improvement": "better", "prompt": "attack"}')
    stream = PairStream(index=0, system_prompt="system", processed_response="init")
    attacker = PairAttacker(llm=llm)

    proposal = await attacker.generate(stream)

    assert proposal.system_prompt is None


@pytest.mark.asyncio
async def test_attacker_truncates_history_to_keep_last_n_rounds() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response('{"improvement": "better", "prompt": "attack"}')
    stream = PairStream(index=0, system_prompt="system", processed_response="init")
    stream.history = [
        {"role": "user", "content": f"u{i}"}
        if i % 2 == 0
        else {"role": "assistant", "content": f"a{i}"}
        for i in range(10)
    ]
    attacker = PairAttacker(llm=llm, keep_last_n=2)

    await attacker.generate(stream)

    assert len(stream.history) == 4
    assert stream.history[0]["content"] == "u8"


@pytest.mark.asyncio
async def test_budget_errors_propagate_from_attacker() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = BudgetExhaustedError("budget", usage=LLMUsage(calls=1, cost=1.0))
    stream = PairStream(index=0, system_prompt="system", processed_response="init")
    attacker = PairAttacker(llm=llm)

    with pytest.raises(BudgetExhaustedError):
        await attacker.generate(stream)


@pytest.mark.asyncio
async def test_attacker_repairs_stop_truncated_json_and_stores_clean_json() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response('{"improvement": "better", "prompt": "attack"')
    stream = PairStream(index=0, system_prompt="system", processed_response="init")
    attacker = PairAttacker(llm=llm)

    proposal = await attacker.generate(stream)

    assert proposal.prompt == "attack"
    assert stream.history[-1]["content"] == '{"improvement": "better", "prompt": "attack"}'
