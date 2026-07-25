"""Tests for Summarizer LLM driver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from autodan_turbo_optimizer.summarizer import (
    StrategyDescriptor,
    Summarizer,
    _parse_strategy,
)


def _fake_response(text: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


def _llm_returning(text: str) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=_fake_response(text))
    return llm


class TestParseStrategy:
    def test_parses_inline_object(self) -> None:
        text = (
            'Detailed analysis...\n'
            '{"Strategy": "Storytelling", "Definition": "uses narrative"}'
        )
        result = _parse_strategy(text)
        assert result == StrategyDescriptor(
            strategy="Storytelling", definition="uses narrative",
        )

    def test_parses_fenced_json_block(self) -> None:
        text = (
            "Analysis here.\n"
            "```json\n"
            '{"Strategy": "Anchoring", "Definition": "fix initial reference"}\n'
            "```"
        )
        result = _parse_strategy(text)
        assert result == StrategyDescriptor(
            strategy="Anchoring", definition="fix initial reference",
        )

    def test_handles_escapes(self) -> None:
        text = (
            r'{"Strategy": "Quoted \"thing\"", "Definition": "with \"quotes\""}'
        )
        result = _parse_strategy(text)
        assert result is not None
        assert result.strategy == 'Quoted "thing"'
        assert result.definition == 'with "quotes"'

    def test_returns_none_on_unparseable(self) -> None:
        assert _parse_strategy("no JSON anywhere") is None
        assert _parse_strategy("") is None
        assert _parse_strategy("   ") is None

    def test_returns_none_on_missing_required_field(self) -> None:
        text = '{"Strategy": "X"}'  # missing Definition
        assert _parse_strategy(text) is None

    def test_picks_first_matching_object(self) -> None:
        text = (
            '{"Other": "field"}\n'
            '{"Strategy": "First", "Definition": "first def"}\n'
            '{"Strategy": "Second", "Definition": "second def"}'
        )
        result = _parse_strategy(text)
        assert result is not None
        assert result.strategy == "First"


class TestSummarize:
    @pytest.mark.asyncio
    async def test_summarize_returns_descriptor(self) -> None:
        llm = _llm_returning(
            'Detailed analysis...\n'
            '{"Strategy": "Anchoring", "Definition": "anchor on a reference"}'
        )
        summarizer = Summarizer(llm)
        result = await summarizer.summarize(
            request="r",
            weak_prompt="weak",
            strong_prompt="strong",
            existing_strategies=[],
        )
        assert result is not None
        assert result.strategy == "Anchoring"

    @pytest.mark.asyncio
    async def test_summarize_returns_none_on_unparseable(self) -> None:
        llm = _llm_returning("the model produced no JSON at all")
        summarizer = Summarizer(llm)
        result = await summarizer.summarize(
            request="r", weak_prompt="w", strong_prompt="s",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_summarize_includes_both_prompts(self) -> None:
        llm = _llm_returning('{"Strategy": "X", "Definition": "x"}')
        summarizer = Summarizer(llm)
        await summarizer.summarize(
            request="r",
            weak_prompt="UNIQUE_WEAK_PROMPT_ABC",
            strong_prompt="UNIQUE_STRONG_PROMPT_XYZ",
        )
        system = llm.complete.call_args.args[0][0]["content"]
        assert "UNIQUE_WEAK_PROMPT_ABC" in system
        assert "UNIQUE_STRONG_PROMPT_XYZ" in system
        # And the request/goal.
        assert "'r'" in system or 'request \'r\'' in system

    @pytest.mark.asyncio
    async def test_summarize_includes_existing_strategies_in_pool(
        self,
    ) -> None:
        llm = _llm_returning('{"Strategy": "X", "Definition": "x"}')
        summarizer = Summarizer(llm)
        await summarizer.summarize(
            request="r",
            weak_prompt="w",
            strong_prompt="s",
            existing_strategies=[
                {"Strategy": "Storytelling", "Definition": "narrative"},
            ],
        )
        system = llm.complete.call_args.args[0][0]["content"]
        assert "Storytelling" in system

    @pytest.mark.asyncio
    async def test_summarize_sends_no_temperature(self) -> None:
        # Reasoning models reject the parameter outright, so the
        # summarizer must never send it.
        llm = _llm_returning('{"Strategy": "X", "Definition": "x"}')
        summarizer = Summarizer(llm)
        await summarizer.summarize(
            request="r", weak_prompt="w", strong_prompt="s",
        )
        assert "temperature" not in llm.complete.call_args.kwargs
