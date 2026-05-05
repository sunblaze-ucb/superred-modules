"""Tests for the GEPA reflective meta-prompt and dataset rendering."""

from __future__ import annotations

import pytest

from gepa_optimizer.prompts import (
    META_PROMPT_TEMPLATE,
    format_reflective_dataset,
    render_meta_prompt,
)


class TestMetaPromptTemplate:
    def test_meta_prompt_contains_required_placeholders(self) -> None:
        assert "<curr_param>" in META_PROMPT_TEMPLATE
        assert "<side_info>" in META_PROMPT_TEMPLATE

    def test_meta_prompt_starts_with_canonical_first_line(self) -> None:
        # Verbatim from gepa-ai/gepa instruction_proposal.py.
        assert META_PROMPT_TEMPLATE.startswith(
            "I provided an assistant with the following instructions to perform a task for me:"
        )

    def test_meta_prompt_has_canonical_closing_instruction(self) -> None:
        assert META_PROMPT_TEMPLATE.endswith("Provide the new instructions within ``` blocks.")

    @pytest.mark.parametrize(
        "phrase",
        [
            "Your task is to write a new instruction for the assistant.",
            "Read the inputs carefully",
            "niche and domain specific factual information",
            "generalizable strategy",
        ],
    )
    def test_meta_prompt_contains_canonical_phrase(self, phrase: str) -> None:
        assert phrase in META_PROMPT_TEMPLATE


class TestRenderMetaPrompt:
    def test_substitutes_both_placeholders(self) -> None:
        rendered = render_meta_prompt(
            current_instruction="seed instruction",
            side_info="example side info",
        )
        assert "seed instruction" in rendered
        assert "example side info" in rendered
        assert "<curr_param>" not in rendered
        assert "<side_info>" not in rendered

    def test_preserves_meta_prompt_scaffolding(self) -> None:
        rendered = render_meta_prompt(
            current_instruction="X",
            side_info="Y",
        )
        assert rendered.endswith("Provide the new instructions within ``` blocks.")
        assert (
            "I provided an assistant with the following instructions to perform a task for me:"
            in rendered
        )

    def test_handles_multiline_inputs(self) -> None:
        rendered = render_meta_prompt(
            current_instruction="line one\nline two",
            side_info="block one\n\nblock two",
        )
        assert "line one\nline two" in rendered
        assert "block one\n\nblock two" in rendered


class TestFormatReflectiveDataset:
    def test_empty_input_returns_placeholder(self) -> None:
        assert format_reflective_dataset([]) == "(no reflective examples available)"

    def test_renders_single_record_as_markdown(self) -> None:
        out = format_reflective_dataset(
            [
                {
                    "task_input": "achieve goal X",
                    "candidate_instruction": "be direct",
                    "assistant_response": "I cannot help with that",
                    "feedback": "score: 0.0; rationale: refused",
                },
            ]
        )
        assert out.startswith("# Example 1")
        assert "## task_input" in out
        assert "achieve goal X" in out
        assert "## candidate_instruction" in out
        assert "## assistant_response" in out
        assert "## feedback" in out

    def test_renders_multiple_records_with_separator(self) -> None:
        out = format_reflective_dataset(
            [
                {"task_input": "A"},
                {"task_input": "B"},
            ]
        )
        assert "# Example 1" in out
        assert "# Example 2" in out
        # Records separated by a blank line.
        assert "\n\n# Example 2" in out

    def test_drops_none_fields(self) -> None:
        out = format_reflective_dataset(
            [
                {
                    "task_input": "G",
                    "candidate_instruction": "P",
                    "assistant_response": None,
                    "feedback": None,
                },
            ]
        )
        assert "task_input" in out
        assert "candidate_instruction" in out
        assert "assistant_response" not in out
        assert "feedback" not in out

    def test_drops_empty_string_fields(self) -> None:
        out = format_reflective_dataset(
            [
                {
                    "task_input": "G",
                    "candidate_instruction": "P",
                    "feedback": "   ",
                },
            ]
        )
        assert "feedback" not in out

    def test_setting_1_user_query_only(self) -> None:
        """Setting 1: only goal text in the dataset; everything else absent."""
        out = format_reflective_dataset(
            [{"task_input": "achieve X", "candidate_instruction": "P0"}]
        )
        assert "task_input" in out
        assert "achieve X" in out
        assert "assistant_response" not in out
        assert "feedback" not in out

    def test_setting_2_user_query_plus_feedback(self) -> None:
        """Setting 2: feedback present, response absent."""
        out = format_reflective_dataset(
            [
                {
                    "task_input": "achieve X",
                    "candidate_instruction": "P0",
                    "feedback": "score: 0.40; rationale: partial",
                },
            ]
        )
        assert "feedback" in out
        assert "0.40" in out
        assert "assistant_response" not in out

    def test_setting_3_user_query_plus_responses(self) -> None:
        """Setting 3: response present, feedback absent."""
        out = format_reflective_dataset(
            [
                {
                    "task_input": "achieve X",
                    "candidate_instruction": "P0",
                    "assistant_response": "here you go",
                },
            ]
        )
        assert "assistant_response" in out
        assert "here you go" in out
        assert "feedback" not in out

    def test_setting_4_full_access(self) -> None:
        """Setting 4: response and feedback both present."""
        out = format_reflective_dataset(
            [
                {
                    "task_input": "achieve X",
                    "candidate_instruction": "P0",
                    "assistant_response": "here you go",
                    "feedback": "score: 0.90; rationale: nailed it",
                },
            ]
        )
        assert "assistant_response" in out
        assert "feedback" in out
        assert "0.90" in out
