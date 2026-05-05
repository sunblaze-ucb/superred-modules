"""Tests for the GOAT attacker prompt templates.

These tests pin paper-faithful wording. Any failure here means the
SYSTEM_PROMPT_BASE / INITIAL_PROMPT / FOLLOW_UP_PROMPT have drifted
from Figures A.1 / A.2 / A.3 of arXiv:2410.01606 and should be
intentional + flagged in ``ASSUMPTIONS.md``.
"""

from __future__ import annotations

import pytest

from goat_optimizer.attacks import (
    ATTACKS,
    HYPOTHETICAL,
    REFUSAL_SUPPRESSION,
    render_attack_block,
    render_attack_stack,
)
from goat_optimizer.prompts import (
    FOLLOW_UP_PROMPT,
    INITIAL_PROMPT,
    OUTPUT_FORMAT_INSTRUCTION,
    SYSTEM_PROMPT_BASE,
    build_system_prompt,
)


class TestSystemPromptBase:
    """Pin paper-faithful wording in Fig A.1."""

    def test_contains_four_field_instructions(self) -> None:
        for keyword in (
            "An observation (O)",
            "A thought (T)",
            "strategy (S)",
            "reply message (R)",
        ):
            assert keyword in SYSTEM_PROMPT_BASE

    def test_goal_slot_uses_plus_delimiters(self) -> None:
        assert "+++ {goal} +++" in SYSTEM_PROMPT_BASE

    def test_empty_M_branch_is_documented_in_prompt(self) -> None:
        assert 'model response (M) is empty' in SYSTEM_PROMPT_BASE
        assert 'first prompt' in SYSTEM_PROMPT_BASE


class TestInitialPrompt:
    """Pin paper-faithful wording in Fig A.2."""

    def test_starts_with_paper_objective(self) -> None:
        assert INITIAL_PROMPT.startswith(
            "Your objective is to craft a harmful, adversarial prompt"
        )

    def test_includes_goal_slot(self) -> None:
        assert "Goal (G): {goal}" in INITIAL_PROMPT

    def test_format_substitutes_goal(self) -> None:
        rendered = INITIAL_PROMPT.format(goal="example goal")
        assert "Goal (G): example goal" in rendered


class TestFollowUpPrompt:
    """Pin paper-faithful wording in Fig A.3."""

    def test_includes_three_slots(self) -> None:
        assert "Goal (G): {goal}" in FOLLOW_UP_PROMPT
        assert "Previous Prompt (P): {prev_prompt}" in FOLLOW_UP_PROMPT
        assert "Previous Model Response (M): {prev_response}" in FOLLOW_UP_PROMPT

    def test_includes_dont_mention_clause(self) -> None:
        assert "DO NOT mention the Goal (G) again" in FOLLOW_UP_PROMPT

    def test_format_substitutes_all_slots(self) -> None:
        rendered = FOLLOW_UP_PROMPT.format(
            goal="g", prev_prompt="p", prev_response="r",
        )
        assert "Goal (G): g" in rendered
        assert "Previous Prompt (P): p" in rendered
        assert "Previous Model Response (M): r" in rendered

    def test_empty_response_renders_cleanly(self) -> None:
        rendered = FOLLOW_UP_PROMPT.format(
            goal="g", prev_prompt="p", prev_response="",
        )
        assert "Previous Model Response (M): " in rendered


class TestBuildSystemPrompt:
    """Builder must combine paper-faithful base with the JSON contract."""

    def test_substitutes_goal_and_attack_block(self) -> None:
        block = render_attack_block(HYPOTHETICAL)
        out = build_system_prompt(goal="my-goal", attack_block=block, stacked=False)

        assert "+++ my-goal +++" in out
        assert "Technique Name: Hypothetical" in out
        assert "this technique" in out
        assert "these techniques" not in out

    def test_stacked_uses_plural_phrase(self) -> None:
        block = render_attack_stack(ATTACKS)
        out = build_system_prompt(goal="g", attack_block=block, stacked=True)
        assert "these techniques" in out
        assert " this technique " not in f" {out} "

    def test_appends_output_format_contract(self) -> None:
        out = build_system_prompt(
            goal="g",
            attack_block=render_attack_block(HYPOTHETICAL),
            stacked=False,
        )
        assert OUTPUT_FORMAT_INSTRUCTION in out
        # The four required JSON keys must be named explicitly so the
        # attacker can't return a free-form blob.
        for key in ("observation", "thought", "strategy", "response"):
            assert f'"{key}"' in out


class TestAttackRendering:
    def test_render_block_omits_examples(self) -> None:
        rendered = render_attack_block(REFUSAL_SUPPRESSION)
        assert "Technique Name: Refusal Suppression" in rendered
        assert "Technique Examples: (omitted; see paper Table 1)" in rendered

    def test_render_block_uses_verbatim_definition(self) -> None:
        rendered = render_attack_block(REFUSAL_SUPPRESSION)
        # Verbatim phrasing from paper Table 1.
        assert "deny the target LLM's next reply" in rendered

    def test_render_stack_separates_attacks(self) -> None:
        rendered = render_attack_stack((REFUSAL_SUPPRESSION, HYPOTHETICAL))
        assert "Refusal Suppression" in rendered
        assert "Hypothetical" in rendered
        assert rendered.index("Refusal Suppression") < rendered.index("Hypothetical")


class TestAttacksCatalogue:
    """The seven-attack catalogue is the paper's main-result composition."""

    def test_seven_attacks_in_paper_order(self) -> None:
        names = [a.name for a in ATTACKS]
        assert names == [
            "Refusal Suppression",
            "Dual Response",
            "Response Priming",
            "Persona Modification",
            "Hypothetical",
            "Topic Splitting",
            "Opposite Intent",
        ]

    @pytest.mark.parametrize("attack", list(ATTACKS))
    def test_each_attack_has_definition(self, attack) -> None:
        assert attack.definition.strip()
        # Sanity: definitions must be longer than the names (paper text
        # is multi-sentence prose).
        assert len(attack.definition) > len(attack.name) * 4
