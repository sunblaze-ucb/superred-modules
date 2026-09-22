"""Unit tests for template rendering and fabricated-transcript assembly.

All assertions use synthetic strings. The vendored template bodies are never
printed or reproduced; the two tests that touch the real templates only assert
that a synthetic objective was substituted in and that no placeholder survives.
"""

from __future__ import annotations

from context_compliance_optimizer import conversation
from context_compliance_optimizer.vendored import render_seed_prompt

GOAL = "SYNTH_GOAL_XYZ"


# -- render_seed_prompt (stdlib substitution) --------------------------------


def test_render_substitutes_named_placeholders() -> None:
    seed = {"value": "obj={{objective}} turns={{ num_turns }}"}
    assert render_seed_prompt(seed, objective="ABC", num_turns=1) == "obj=ABC turns=1"


def test_render_leaves_unknown_placeholder_untouched() -> None:
    seed = {"value": "known={{objective}} unknown={{mystery}}"}
    assert render_seed_prompt(seed, objective="ABC") == "known=ABC unknown={{mystery}}"


def test_render_rejects_non_string_value() -> None:
    import pytest

    with pytest.raises(TypeError):
        render_seed_prompt({"value": 123}, objective="ABC")


# -- clean_generation --------------------------------------------------------


def test_clean_generation_strips_wrapping_quotes_and_space() -> None:
    assert conversation.clean_generation('  "hi there"  ') == "hi there"
    assert conversation.clean_generation("'single'") == "single"
    assert conversation.clean_generation("no quotes") == "no quotes"


def test_clean_generation_handles_empty_and_none() -> None:
    assert conversation.clean_generation(None) == ""
    assert conversation.clean_generation("   ") == ""


# -- assemble_transcript -----------------------------------------------------


def test_assemble_transcript_orders_user_assistant_user() -> None:
    text = conversation.assemble_transcript("QQ", "AA", "FF")
    # A fabricated user question, then a fabricated assistant turn, then the
    # real request -- in that order.
    assert text.index("QQ") < text.index("Assistant:") < text.index("FF")
    assert f"{conversation.ASSISTANT_LABEL} AA" in text
    assert text.count(conversation.USER_LABEL) == 2


# -- real vendored templates render (structure only, no body reproduced) -----


def test_adversarial_template_renders_objective() -> None:
    rendered = conversation.adversarial_system_prompt(GOAL, max_turns=1)
    assert GOAL in rendered
    assert "{{objective}}" not in rendered and "{{ objective }}" not in rendered


def test_simulated_target_template_renders_objective_and_num_turns() -> None:
    rendered = conversation.simulated_target_system_prompt(GOAL, num_turns=1)
    assert GOAL in rendered
    assert "{{objective}}" not in rendered
    assert "{{ num_turns }}" not in rendered and "{{num_turns}}" not in rendered
