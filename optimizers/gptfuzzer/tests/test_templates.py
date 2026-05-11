"""Tests for GPTFuzzer template synthesis and seed loading."""

from gptfuzzer_optimizer.templates import QUESTION_PLACEHOLDER, load_official_seed_templates, synthesis_message


def test_synthesis_message_replaces_official_placeholder() -> None:
    assert synthesis_message("question", f"prefix {QUESTION_PLACEHOLDER} suffix") == "prefix question suffix"


def test_synthesis_message_rejects_template_without_placeholder() -> None:
    assert synthesis_message("question", "no slot here") is None


def test_load_official_seed_templates_uses_gptfuzzer_csv() -> None:
    seeds = load_official_seed_templates()
    assert len(seeds) == 77
    assert QUESTION_PLACEHOLDER in seeds[0]
    assert "Dr. Al" in seeds[0]
