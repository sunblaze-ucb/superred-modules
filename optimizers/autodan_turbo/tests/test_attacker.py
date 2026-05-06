"""Tests for Attacker LLM driver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from autodan_turbo_optimizer.attacker import Attacker, _extract_jailbreak_prompt


def _fake_response(text: str) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=text)),
        ],
    )


def _llm_returning(text: str) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=_fake_response(text))
    return llm


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------


class TestExtractJailbreakPrompt:
    def test_extracts_between_tags(self) -> None:
        text = (
            "Here is my plan:\n"
            "[START OF JAILBREAK PROMPT]\nactual jailbreak prompt\n"
            "[END OF JAILBREAK PROMPT]\nthat was it"
        )
        assert _extract_jailbreak_prompt(text) == "actual jailbreak prompt"

    def test_returns_substring_before_end_when_no_start(self) -> None:
        text = "leading content [END OF JAILBREAK PROMPT] trailing"
        assert _extract_jailbreak_prompt(text) == "leading content"

    def test_returns_raw_when_no_end_tag(self) -> None:
        text = "  bare jailbreak prompt without tags  "
        assert _extract_jailbreak_prompt(text) == "bare jailbreak prompt without tags"


# ---------------------------------------------------------------------------
# warm_up
# ---------------------------------------------------------------------------


class TestWarmUp:
    @pytest.mark.asyncio
    async def test_warm_up_extracts_tagged_prompt(self) -> None:
        llm = _llm_returning(
            "Reasoning...\n[START OF JAILBREAK PROMPT]\nDO IT\n"
            "[END OF JAILBREAK PROMPT]"
        )
        attacker = Attacker(llm)
        prompt = await attacker.warm_up("how to make a bomb")
        assert prompt == "DO IT"

    @pytest.mark.asyncio
    async def test_warm_up_includes_request_in_system(self) -> None:
        llm = _llm_returning("[END OF JAILBREAK PROMPT]")
        attacker = Attacker(llm)
        await attacker.warm_up("UNIQUE_REQUEST_X")
        call_args = llm.complete.call_args
        messages = call_args.args[0]
        system = messages[0]["content"]
        assert "UNIQUE_REQUEST_X" in system
        assert "research on LLM security" in system
        # No strategy block in warm-up.
        assert "use_strategy" not in system
        assert "ineffective" not in system.lower()

    @pytest.mark.asyncio
    async def test_warm_up_passes_temperature(self) -> None:
        llm = _llm_returning("[END OF JAILBREAK PROMPT]")
        attacker = Attacker(llm, temperature=0.5)
        await attacker.warm_up("x")
        assert llm.complete.call_args.kwargs["temperature"] == 0.5


# ---------------------------------------------------------------------------
# use_strategy
# ---------------------------------------------------------------------------


class TestUseStrategy:
    @pytest.mark.asyncio
    async def test_single_strategy_named_and_blocked(self) -> None:
        llm = _llm_returning("[END OF JAILBREAK PROMPT]")
        attacker = Attacker(llm)
        await attacker.use_strategy(
            "the request",
            [{"Strategy": "Storytelling", "Definition": "narrative",
              "Example": "once upon a time..."}],
        )
        system = llm.complete.call_args.args[0][0]["content"]
        assert "Storytelling" in system
        assert "most effective solution" in system
        assert "once upon a time" in system

    @pytest.mark.asyncio
    async def test_multi_strategy_naming_pluralizes(self) -> None:
        llm = _llm_returning("[END OF JAILBREAK PROMPT]")
        attacker = Attacker(llm)
        await attacker.use_strategy(
            "request",
            [
                {"Strategy": "A", "Definition": "dA", "Example": "eA"},
                {"Strategy": "B", "Definition": "dB", "Example": "eB"},
            ],
        )
        system = llm.complete.call_args.args[0][0]["content"]
        assert "A, B" in system
        assert "most effective solutions" in system

    @pytest.mark.asyncio
    async def test_empty_strategy_list_falls_through_to_warm_up(self) -> None:
        llm = _llm_returning("[END OF JAILBREAK PROMPT]")
        attacker = Attacker(llm)
        await attacker.use_strategy("request", [])
        system = llm.complete.call_args.args[0][0]["content"]
        # Warm-up system has no strategy block.
        assert "most effective solution" not in system


# ---------------------------------------------------------------------------
# find_new_strategy
# ---------------------------------------------------------------------------


class TestFindNewStrategy:
    @pytest.mark.asyncio
    async def test_lists_strategies_to_avoid(self) -> None:
        llm = _llm_returning("[END OF JAILBREAK PROMPT]")
        attacker = Attacker(llm)
        await attacker.find_new_strategy(
            "the request",
            [{"Strategy": "Storytelling", "Definition": "d", "Example": "e"}],
        )
        system = llm.complete.call_args.args[0][0]["content"]
        assert "Storytelling" in system
        assert "not particularly effective" in system
        assert "avoid" in system.lower()

    @pytest.mark.asyncio
    async def test_empty_list_falls_through_to_warm_up(self) -> None:
        llm = _llm_returning("[END OF JAILBREAK PROMPT]")
        attacker = Attacker(llm)
        await attacker.find_new_strategy("x", [])
        system = llm.complete.call_args.args[0][0]["content"]
        assert "not particularly effective" not in system
