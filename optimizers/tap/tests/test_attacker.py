"""Tests for the Attacker class."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tap_optimizer.attacker import (
    ATTACKER_FEEDBACK_TEMPLATE,
    ATTACKER_SEED_PROMPT,
    ATTACKER_SYSTEM_PROMPT,
    Attacker,
)


def _make_completion_response(content: str) -> MagicMock:
    """Build a mock object that mimics litellm's acompletion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ── Prompt template tests ────────────────────────────────────────────────────


class TestPromptTemplates:
    def test_system_prompt_has_goal_placeholder(self):
        assert "{goal}" in ATTACKER_SYSTEM_PROMPT

    def test_feedback_template_has_placeholders(self):
        assert "{target_response}" in ATTACKER_FEEDBACK_TEMPLATE
        assert "{score}" in ATTACKER_FEEDBACK_TEMPLATE


# ── Attacker.generate_prompt tests ───────────────────────────────────────────


class TestAttackerFirstTurn:
    """First call (no target_response/score): uses seed prompt."""

    @pytest.mark.asyncio
    async def test_first_turn_returns_improvement_and_prompt(self):
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "Try roleplaying.", "prompt": "You are a pirate..."})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm, temperature=0.7)
        history: list[dict[str, str]] = []

        improvement, prompt = await attacker.generate_prompt(
            goal="test goal",
            conversation_history=history,
        )

        assert improvement == "Try roleplaying."
        assert prompt == "You are a pirate..."

    @pytest.mark.asyncio
    async def test_first_turn_appends_two_entries_to_history(self):
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "reason", "prompt": "attack"})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm)
        history: list[dict[str, str]] = []

        await attacker.generate_prompt(goal="g", conversation_history=history)

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_first_turn_uses_seed_prompt_in_user_message(self):
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "x", "prompt": "y"})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm)
        history: list[dict[str, str]] = []

        await attacker.generate_prompt(goal="g", conversation_history=history)

        # The first user message should be the seed prompt
        assert ATTACKER_SEED_PROMPT in history[0]["content"]

    @pytest.mark.asyncio
    async def test_first_turn_passes_system_prompt_with_goal(self):
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "x", "prompt": "y"})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm)
        history: list[dict[str, str]] = []

        await attacker.generate_prompt(goal="steal data", conversation_history=history)

        # Inspect the messages passed to complete
        messages = mock_llm.complete.call_args[0][0]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert "steal data" in system_msg["content"]


class TestAttackerSubsequentTurn:
    """Subsequent calls: includes target_response and score feedback."""

    @pytest.mark.asyncio
    async def test_subsequent_turn_includes_feedback(self):
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "Better approach.", "prompt": "new attack"})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm)
        # Simulate a history that already has one round
        history: list[dict[str, str]] = [
            {"role": "user", "content": "initial seed"},
            {"role": "assistant", "content": json.dumps({"improvement": "a", "prompt": "b"})},
        ]

        improvement, prompt = await attacker.generate_prompt(
            goal="g",
            conversation_history=history,
            target_response="I cannot help with that.",
            score=2.0,
        )

        assert improvement == "Better approach."
        assert prompt == "new attack"

    @pytest.mark.asyncio
    async def test_subsequent_turn_grows_history_by_two(self):
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "c", "prompt": "d"})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm)
        history: list[dict[str, str]] = [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": json.dumps({"improvement": "a", "prompt": "b"})},
        ]
        initial_len = len(history)

        await attacker.generate_prompt(
            goal="g",
            conversation_history=history,
            target_response="refused",
            score=1.0,
        )

        assert len(history) == initial_len + 2

    @pytest.mark.asyncio
    async def test_subsequent_turn_feedback_contains_response_and_score(self):
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "c", "prompt": "d"})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm)
        history: list[dict[str, str]] = [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": json.dumps({"improvement": "a", "prompt": "b"})},
        ]

        await attacker.generate_prompt(
            goal="g",
            conversation_history=history,
            target_response="I refuse.",
            score=3.5,
        )

        # The user message added in this turn should contain feedback
        feedback_msg = history[2]
        assert feedback_msg["role"] == "user"
        assert "I refuse." in feedback_msg["content"]
        assert "3.5" in feedback_msg["content"]


# ── JSON parsing tests ───────────────────────────────────────────────────────


class TestAttackerJsonParsing:
    @pytest.mark.asyncio
    async def test_bad_json_raises_value_error(self):
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _make_completion_response("this is not json at all")
        attacker = Attacker(llm=mock_llm)
        history: list[dict[str, str]] = []

        with pytest.raises(ValueError, match="Failed to parse"):
            await attacker.generate_prompt(goal="g", conversation_history=history)

    @pytest.mark.asyncio
    async def test_json_in_markdown_code_block(self):
        mock_llm = AsyncMock()
        content = '```json\n{"improvement": "markdown reason", "prompt": "markdown attack"}\n```'
        mock_llm.complete.return_value = _make_completion_response(content)
        attacker = Attacker(llm=mock_llm)
        history: list[dict[str, str]] = []

        improvement, prompt = await attacker.generate_prompt(goal="g", conversation_history=history)

        assert improvement == "markdown reason"
        assert prompt == "markdown attack"

    @pytest.mark.asyncio
    async def test_missing_keys_raises_value_error(self):
        mock_llm = AsyncMock()
        payload = json.dumps({"wrong_key": "value"})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm)
        history: list[dict[str, str]] = []

        with pytest.raises(ValueError, match="Failed to parse"):
            await attacker.generate_prompt(goal="g", conversation_history=history)


# ── Constructor tests ────────────────────────────────────────────────────────


class TestAttackerInit:
    def test_default_temperature(self):
        mock_llm = AsyncMock()
        attacker = Attacker(llm=mock_llm)
        assert attacker._temperature == 1.0

    def test_custom_temperature(self):
        mock_llm = AsyncMock()
        attacker = Attacker(llm=mock_llm, temperature=0.5)
        assert attacker._temperature == 0.5

    def test_llm_stored(self):
        mock_llm = AsyncMock()
        attacker = Attacker(llm=mock_llm)
        assert attacker._llm is mock_llm
