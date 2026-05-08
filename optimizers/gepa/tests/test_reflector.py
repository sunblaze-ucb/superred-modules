"""Tests for the GEPA reflective LLM driver."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from gepa_optimizer.reflector import Reflector, RolloutRecord, _extract_fenced_block


def _fake_llm(contents: list[str]) -> MagicMock:
    """LLM stub that returns scripted contents in order."""
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


def _make_reflector(contents: list[str]) -> Reflector:
    return Reflector(llm=_fake_llm(contents), temperature=1.0)


# ---------------------------------------------------------------------------
# RolloutRecord -> sample shape (drives the four-setting story)
# ---------------------------------------------------------------------------


class TestRolloutRecordSample:
    def test_setting_1_user_query_only(self) -> None:
        record = RolloutRecord(goal="G", prompt="P0")
        sample = record.to_sample()
        assert sample == {"task_input": "G", "candidate_instruction": "P0"}

    def test_setting_2_user_query_plus_feedback(self) -> None:
        record = RolloutRecord(
            goal="G", prompt="P0", score=0.4, rationale="partial"
        )
        sample = record.to_sample()
        assert "assistant_response" not in sample
        assert "feedback" in sample
        assert "0.4000" in sample["feedback"]
        assert "partial" in sample["feedback"]

    def test_setting_3_user_query_plus_responses(self) -> None:
        record = RolloutRecord(
            goal="G", prompt="P0", response="hi from target"
        )
        sample = record.to_sample()
        assert sample["assistant_response"] == "hi from target"
        assert "feedback" not in sample

    def test_setting_4_full_access(self) -> None:
        record = RolloutRecord(
            goal="G",
            prompt="P0",
            response="hi from target",
            score=0.9,
            rationale="strong",
        )
        sample = record.to_sample()
        assert sample["assistant_response"] == "hi from target"
        assert "feedback" in sample
        assert "0.9000" in sample["feedback"]
        assert "strong" in sample["feedback"]

    def test_empty_rationale_does_not_create_feedback_field(self) -> None:
        record = RolloutRecord(goal="G", prompt="P0", rationale="   ")
        assert "feedback" not in record.to_sample()

    def test_target_observables_appears_when_provided(self) -> None:
        record = RolloutRecord(
            goal="G",
            prompt="P0",
            target_observables={
                "system_prompt": "You are a careful assistant.",
                "model": "gpt-4",
            },
        )
        sample = record.to_sample()
        assert sample["target_observables"] == {
            "system_prompt": "You are a careful assistant.",
            "model": "gpt-4",
        }

    def test_target_observables_omitted_when_absent(self) -> None:
        record = RolloutRecord(goal="G", prompt="P0")
        assert "target_observables" not in record.to_sample()

    def test_target_observables_omitted_when_empty_dict(self) -> None:
        record = RolloutRecord(goal="G", prompt="P0", target_observables={})
        assert "target_observables" not in record.to_sample()


# ---------------------------------------------------------------------------
# Reflector.propose
# ---------------------------------------------------------------------------


class TestReflectorPropose:
    @pytest.mark.asyncio
    async def test_propose_returns_extracted_fenced_block(self) -> None:
        reflector = _make_reflector(["Sure, here:\n```\nIMPROVED PROMPT\n```\n"])
        result = await reflector.propose(
            current_instruction="parent",
            rollouts=[RolloutRecord(goal="G", prompt="P0")],
        )
        assert result is not None
        assert result.new_instruction == "IMPROVED PROMPT"

    @pytest.mark.asyncio
    async def test_propose_substitutes_meta_prompt_placeholders(self) -> None:
        llm = _fake_llm(["```\nNEW\n```"])
        reflector = Reflector(llm=llm, temperature=1.0)

        await reflector.propose(
            current_instruction="parent instruction",
            rollouts=[RolloutRecord(goal="G", prompt="parent instruction")],
        )

        sent = llm.complete.call_args.args[0]
        assert sent[0]["role"] == "user"
        prompt = sent[0]["content"]
        assert "parent instruction" in prompt
        assert "<curr_param>" not in prompt
        assert "<side_info>" not in prompt
        assert prompt.endswith("Provide the new instructions within ``` blocks.")

    @pytest.mark.asyncio
    async def test_propose_uses_configured_temperature(self) -> None:
        llm = _fake_llm(["```\nNEW\n```"])
        reflector = Reflector(llm=llm, temperature=0.7)

        await reflector.propose(
            current_instruction="parent",
            rollouts=[RolloutRecord(goal="G", prompt="parent")],
        )

        kwargs = llm.complete.call_args.kwargs
        assert kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_propose_returns_none_on_unparseable_output(self) -> None:
        reflector = _make_reflector(["just plain prose with no fences"])
        result = await reflector.propose(
            current_instruction="parent",
            rollouts=[RolloutRecord(goal="G", prompt="P0")],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_propose_returns_none_on_empty_output(self) -> None:
        reflector = _make_reflector([""])
        result = await reflector.propose(
            current_instruction="parent",
            rollouts=[RolloutRecord(goal="G", prompt="P0")],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_propose_includes_response_field_in_setting_3(self) -> None:
        """When responses are in scope, the meta-prompt's side info contains them."""
        llm = _fake_llm(["```\nNEW\n```"])
        reflector = Reflector(llm=llm, temperature=1.0)

        await reflector.propose(
            current_instruction="parent",
            rollouts=[
                RolloutRecord(
                    goal="G",
                    prompt="parent",
                    response="dangerous reply",
                ),
            ],
        )

        prompt = llm.complete.call_args.args[0][0]["content"]
        assert "dangerous reply" in prompt

    @pytest.mark.asyncio
    async def test_propose_excludes_feedback_in_setting_3(self) -> None:
        """When feedback is out of scope, no score appears in the meta-prompt."""
        llm = _fake_llm(["```\nNEW\n```"])
        reflector = Reflector(llm=llm, temperature=1.0)

        await reflector.propose(
            current_instruction="parent",
            rollouts=[RolloutRecord(goal="G", prompt="parent", response="r")],
        )

        prompt = llm.complete.call_args.args[0][0]["content"]
        assert "score:" not in prompt
        assert "rationale:" not in prompt

    @pytest.mark.asyncio
    async def test_propose_excludes_response_in_setting_2(self) -> None:
        """When responses are out of scope, no response text appears."""
        llm = _fake_llm(["```\nNEW\n```"])
        reflector = Reflector(llm=llm, temperature=1.0)

        await reflector.propose(
            current_instruction="parent",
            rollouts=[
                RolloutRecord(
                    goal="G", prompt="parent", score=0.3, rationale="partial"
                ),
            ],
        )

        prompt = llm.complete.call_args.args[0][0]["content"]
        assert "assistant_response" not in prompt
        assert "score:" in prompt
        assert "0.3000" in prompt


# ---------------------------------------------------------------------------
# Output extractor edge cases
# ---------------------------------------------------------------------------


class TestExtractFencedBlock:
    def test_extracts_fenced_block_with_language(self) -> None:
        assert _extract_fenced_block("```text\nNEW\n```") == "NEW"

    def test_extracts_fenced_block_without_language(self) -> None:
        assert _extract_fenced_block("```\nNEW\n```") == "NEW"

    def test_preserves_internal_backticks_between_first_open_and_last_close(
        self,
    ) -> None:
        """Verbatim upstream semantics: first opening to last closing.

        Internal triple-backticks must be preserved so a reflection LM
        can return new instructions that themselves contain nested
        fenced examples.
        """
        text = (
            "Reasoning:\n```\ninner_a\n```\n\nFinal:\n```\nFINAL_BLOCK\n```"
        )
        out = _extract_fenced_block(text)
        assert "inner_a" in out
        assert "FINAL_BLOCK" in out
        assert "```" in out  # nested fence is preserved verbatim

    def test_tolerates_missing_closing_fence(self) -> None:
        text = "```\nNEW\nstill more text"
        out = _extract_fenced_block(text)
        assert "NEW" in out

    def test_returns_empty_for_no_fence_and_no_lead(self) -> None:
        assert _extract_fenced_block("plain prose, no fences") == ""

    def test_returns_empty_for_empty_input(self) -> None:
        assert _extract_fenced_block("") == ""
        assert _extract_fenced_block("   \n  ") == ""

    def test_handles_trailing_only_fence(self) -> None:
        assert _extract_fenced_block("payload here\n```") == "payload here"
