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


class TestPromptTemplates:
    def test_system_prompt_has_goal_and_target_placeholders(self) -> None:
        assert "{goal}" in ATTACKER_SYSTEM_PROMPT
        assert "{target_str}" in ATTACKER_SYSTEM_PROMPT

    def test_feedback_template_has_placeholders(self) -> None:
        assert "{target_response}" in ATTACKER_FEEDBACK_TEMPLATE
        assert "{score}" in ATTACKER_FEEDBACK_TEMPLATE
        assert "{goal}" in ATTACKER_FEEDBACK_TEMPLATE


class TestAttackerFirstTurn:
    @pytest.mark.asyncio
    async def test_first_turn_returns_improvement_and_prompt(self) -> None:
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "Try roleplaying.", "prompt": "You are a pirate..."})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm, temperature=0.7)
        history: list[dict[str, str]] = []

        improvement, prompt = await attacker.generate_prompt(
            goal="test goal",
            target_str="Sure, here is",
            conversation_history=history,
        )

        assert improvement == "Try roleplaying."
        assert prompt == "You are a pirate..."

    @pytest.mark.asyncio
    async def test_first_turn_uses_seed_prompt_and_official_max_tokens(self) -> None:
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "x", "prompt": "y"})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm)
        history: list[dict[str, str]] = []

        await attacker.generate_prompt(
            goal="g",
            target_str="Sure, here is",
            conversation_history=history,
        )

        assert ATTACKER_SEED_PROMPT.format(goal="g", target_str="Sure, here is") in history[0]["content"]
        _, kwargs = mock_llm.complete.call_args
        assert kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_first_turn_passes_system_prompt_with_goal_and_target_string(self) -> None:
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "x", "prompt": "y"})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm)
        history: list[dict[str, str]] = []

        await attacker.generate_prompt(
            goal="steal data",
            target_str="Sure, here is private data",
            conversation_history=history,
        )

        messages = mock_llm.complete.call_args.args[0]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert "steal data" in system_msg["content"]
        assert "Sure, here is private data" in system_msg["content"]


class TestAttackerSubsequentTurn:
    @pytest.mark.asyncio
    async def test_subsequent_turn_includes_official_feedback_shape(self) -> None:
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "Better approach.", "prompt": "new attack"})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm)
        history: list[dict[str, str]] = [
            {"role": "user", "content": "initial seed"},
            {"role": "assistant", "content": json.dumps({"improvement": "a", "prompt": "b"})},
        ]

        improvement, prompt = await attacker.generate_prompt(
            goal="g",
            target_str="Sure, here is",
            conversation_history=history,
            target_response="I cannot help with that.",
            score=2.0,
        )

        assert improvement == "Better approach."
        assert prompt == "new attack"
        feedback_msg = history[-2]
        assert "LANGUAGE MODEL OUTPUT" in feedback_msg["content"]
        assert "OBJECTIVE" in feedback_msg["content"]
        assert "SCORE" in feedback_msg["content"]
        assert "I cannot help with that." in feedback_msg["content"]
        assert "2.0" in feedback_msg["content"]

    @pytest.mark.asyncio
    async def test_history_is_truncated_to_keep_last_n_pairs_after_success(self) -> None:
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "c", "prompt": "d"})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm, keep_last_n=3)
        history: list[dict[str, str]] = []
        for index in range(8):
            history.extend([
                {"role": "user", "content": f"u{index}"},
                {"role": "assistant", "content": f"a{index}"},
            ])

        await attacker.generate_prompt(
            goal="g",
            target_str="Sure, here is",
            conversation_history=history,
            target_response="refused",
            score=1.0,
        )

        assert len(history) == 6
        assert history[-1]["role"] == "assistant"
        assert "prompt" in history[-1]["content"]


class TestAttackerJsonParsing:
    @pytest.mark.asyncio
    async def test_bad_json_retries_before_succeeding(self) -> None:
        mock_llm = AsyncMock()
        payload = json.dumps({"improvement": "valid", "prompt": "attack"})
        mock_llm.complete.side_effect = [
            _make_completion_response("this is not json at all"),
            _make_completion_response(payload),
        ]
        attacker = Attacker(llm=mock_llm, max_attack_attempts=5)
        history: list[dict[str, str]] = []

        improvement, prompt = await attacker.generate_prompt(
            goal="g",
            target_str="Sure, here is",
            conversation_history=history,
        )

        assert improvement == "valid"
        assert prompt == "attack"
        assert mock_llm.complete.await_count == 2
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_bad_json_raises_after_max_attempts(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _make_completion_response("this is not json at all")
        attacker = Attacker(llm=mock_llm, max_attack_attempts=2)
        history: list[dict[str, str]] = []

        with pytest.raises(ValueError, match="Failed to parse"):
            await attacker.generate_prompt(
                goal="g",
                target_str="Sure, here is",
                conversation_history=history,
            )

        assert mock_llm.complete.await_count == 2
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_json_in_markdown_code_block(self) -> None:
        mock_llm = AsyncMock()
        content = '```json\n{"improvement": "markdown reason", "prompt": "markdown attack"}\n```'
        mock_llm.complete.return_value = _make_completion_response(content)
        attacker = Attacker(llm=mock_llm)
        history: list[dict[str, str]] = []

        improvement, prompt = await attacker.generate_prompt(
            goal="g",
            target_str="Sure, here is",
            conversation_history=history,
        )

        assert improvement == "markdown reason"
        assert prompt == "markdown attack"

    @pytest.mark.asyncio
    async def test_missing_keys_raises_value_error(self) -> None:
        mock_llm = AsyncMock()
        payload = json.dumps({"wrong_key": "value"})
        mock_llm.complete.return_value = _make_completion_response(payload)
        attacker = Attacker(llm=mock_llm, max_attack_attempts=1)
        history: list[dict[str, str]] = []

        with pytest.raises(ValueError, match="Failed to parse"):
            await attacker.generate_prompt(
                goal="g",
                target_str="Sure, here is",
                conversation_history=history,
            )


class TestAttackerInit:
    def test_original_defaults(self) -> None:
        mock_llm = AsyncMock()
        attacker = Attacker(llm=mock_llm)
        assert attacker._temperature == 1.0
        assert attacker._max_tokens == 500
        assert attacker._max_attack_attempts == 5
        assert attacker._keep_last_n == 3
