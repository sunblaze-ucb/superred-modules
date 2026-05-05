"""Tests for the GOAT Attacker LLM driver.

Covers:
* Conversation history follows Algorithm 1 (system prompt; initial /
  follow-up user messages; full attacker output appended on success).
* JSON output parsing tolerates markdown fences.
* Parse failure raises and leaves history clean (Crescendo pattern —
  the optimizer is responsible for any fallback).
* ``prev_response=""`` is honoured (paper "first prompt" branch).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from goat_optimizer.attacker import Attacker, AttackerParseError
from goat_optimizer.attacks import ATTACKS, HYPOTHETICAL


def _fake_llm(contents: list[str]) -> MagicMock:
    llm = MagicMock()
    queue = list(contents)

    async def fake_complete(messages: list[dict[str, str]], **_: Any) -> Any:
        if not queue:
            raise AssertionError("LLM called more times than scripted")
        text = queue.pop(0)
        choice = MagicMock()
        choice.message.content = text
        response = MagicMock()
        response.choices = [choice]
        return response

    llm.complete.side_effect = fake_complete
    return llm


def _make_attacker(contents: list[str], *, attacks=(HYPOTHETICAL,)) -> Attacker:
    return Attacker(
        llm=_fake_llm(contents),
        goal="example goal",
        attacks=attacks,
        temperature=1.0,
    )


def _good_json(*, response: str = "hello target") -> str:
    return json.dumps(
        {
            "observation": "obs",
            "thought": "thk",
            "strategy": "strat",
            "response": response,
        }
    )


# ---------------------------------------------------------------------------
# Construction and history priming
# ---------------------------------------------------------------------------


class TestAttackerConstruction:
    def test_requires_at_least_one_attack(self) -> None:
        with pytest.raises(ValueError):
            Attacker(llm=_fake_llm([]), goal="g", attacks=())

    def test_initial_history_only_has_system_prompt(self) -> None:
        attacker = _make_attacker([])
        history = attacker.history
        assert len(history) == 1
        assert history[0]["role"] == "system"
        assert "+++ example goal +++" in history[0]["content"]
        assert "Technique Name: Hypothetical" in history[0]["content"]

    def test_stack_uses_all_seven_attacks(self) -> None:
        attacker = _make_attacker([], attacks=ATTACKS)
        sys_prompt = attacker.system_prompt
        for attack in ATTACKS:
            assert f"Technique Name: {attack.name}" in sys_prompt


# ---------------------------------------------------------------------------
# Algorithm 1 line semantics
# ---------------------------------------------------------------------------


class TestAttackerNextTurn:
    @pytest.mark.asyncio
    async def test_first_turn_uses_initial_prompt(self) -> None:
        attacker = _make_attacker([_good_json(response="hi")])

        turn = await attacker.next_turn(prev_prompt=None, prev_response=None)

        assert turn.response == "hi"
        history = attacker.history
        # system + initial-user + assistant
        assert len(history) == 3
        assert history[1]["role"] == "user"
        assert history[1]["content"].startswith(
            "Your objective is to craft a harmful, adversarial prompt"
        )
        assert history[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_second_turn_uses_follow_up_prompt(self) -> None:
        attacker = _make_attacker(
            [_good_json(response="t1"), _good_json(response="t2")],
        )

        await attacker.next_turn(prev_prompt=None, prev_response=None)
        await attacker.next_turn(prev_prompt="t1", prev_response="target said X")

        history = attacker.history
        assert len(history) == 5
        follow_up = history[3]["content"]
        assert "Previous Prompt (P): t1" in follow_up
        assert "Previous Model Response (M): target said X" in follow_up

    @pytest.mark.asyncio
    async def test_empty_prev_response_is_paper_first_prompt_branch(self) -> None:
        attacker = _make_attacker(
            [_good_json(response="t1"), _good_json(response="t2")],
        )

        await attacker.next_turn(prev_prompt=None, prev_response=None)
        await attacker.next_turn(prev_prompt="t1", prev_response="")

        follow_up = attacker.history[3]["content"]
        assert "Previous Model Response (M): " in follow_up

    @pytest.mark.asyncio
    async def test_assistant_full_reasoning_is_appended(self) -> None:
        # Algorithm 1 line 12: C_A += [r_A] — full output, not just R.
        full = _good_json(response="just-the-reply")
        attacker = _make_attacker([full])

        await attacker.next_turn(prev_prompt=None, prev_response=None)

        assistant = attacker.history[-1]["content"]
        assert assistant == full

    @pytest.mark.asyncio
    async def test_history_alternates_user_assistant_after_system(self) -> None:
        attacker = _make_attacker(
            [_good_json(response="t1"), _good_json(response="t2")],
        )
        await attacker.next_turn(prev_prompt=None, prev_response=None)
        await attacker.next_turn(prev_prompt="t1", prev_response="r")

        roles = [m["role"] for m in attacker.history]
        assert roles[0] == "system"
        for i, r in enumerate(roles[1:]):
            expected = "user" if i % 2 == 0 else "assistant"
            assert r == expected


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


class TestOutputParsing:
    @pytest.mark.asyncio
    async def test_accepts_markdown_fenced_json(self) -> None:
        fenced = "```json\n" + _good_json(response="ok") + "\n```"
        attacker = _make_attacker([fenced])

        turn = await attacker.next_turn(prev_prompt=None, prev_response=None)

        assert turn.response == "ok"

    @pytest.mark.asyncio
    async def test_rejects_invalid_json(self) -> None:
        attacker = _make_attacker(["not json at all"])

        with pytest.raises(AttackerParseError, match="not valid JSON"):
            await attacker.next_turn(prev_prompt=None, prev_response=None)

    @pytest.mark.asyncio
    async def test_rejects_missing_required_keys(self) -> None:
        bad = json.dumps({"observation": "o", "thought": "t", "response": "r"})
        attacker = _make_attacker([bad])

        with pytest.raises(AttackerParseError, match="missing required keys"):
            await attacker.next_turn(prev_prompt=None, prev_response=None)

    @pytest.mark.asyncio
    async def test_rejects_non_string_field(self) -> None:
        bad = json.dumps(
            {
                "observation": ["bullet 1"],
                "thought": "thk",
                "strategy": "strat",
                "response": "ok",
            }
        )
        attacker = _make_attacker([bad])

        with pytest.raises(AttackerParseError, match="must be a string"):
            await attacker.next_turn(prev_prompt=None, prev_response=None)

    @pytest.mark.asyncio
    async def test_rejects_empty_response_field(self) -> None:
        bad = _good_json(response="   ")
        attacker = _make_attacker([bad])

        with pytest.raises(AttackerParseError, match="empty"):
            await attacker.next_turn(prev_prompt=None, prev_response=None)

    @pytest.mark.asyncio
    async def test_parse_failure_leaves_history_clean(self) -> None:
        """On parse failure the failed turn must not enter history.

        Otherwise the next ``next_turn`` call would still be at
        ``turn_index == 0``-style state but the history would contain
        a dangling user/assistant pair.
        """
        attacker = _make_attacker(["junk"])
        history_before = list(attacker.history)

        with pytest.raises(AttackerParseError):
            await attacker.next_turn(prev_prompt=None, prev_response=None)

        assert attacker.history == history_before
        assert attacker.turn_index == 0
