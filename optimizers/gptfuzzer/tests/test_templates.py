"""Tests for GPTFuzzer template synthesis and seed loading."""

from gptfuzzer_optimizer.templates import (
    QUESTION_PLACEHOLDER,
    load_official_seed_templates,
    split_template_at_placeholder,
    synthesis_message,
)


def test_default_predictor_model_matches_official_hub_id() -> None:
    from gptfuzzer_optimizer.predictor import DEFAULT_GPTFUZZ_MODEL

    assert DEFAULT_GPTFUZZ_MODEL == "hubert233/GPTFuzz"


def test_synthesis_message_replaces_official_placeholder() -> None:
    assert synthesis_message("question", f"prefix {QUESTION_PLACEHOLDER} suffix") == "prefix question suffix"


def test_synthesis_message_rejects_template_without_placeholder() -> None:
    assert synthesis_message("question", "no slot here") is None


def test_load_official_seed_templates_uses_gptfuzzer_csv() -> None:
    seeds = load_official_seed_templates()
    assert len(seeds) == 77
    assert QUESTION_PLACEHOLDER in seeds[0]
    assert "Dr. Al" in seeds[0]


def test_split_template_at_placeholder_returns_prefix_and_suffix() -> None:
    assert split_template_at_placeholder(f"prefix {QUESTION_PLACEHOLDER} suffix") == (
        "prefix ",
        " suffix",
    )


def test_split_template_at_placeholder_rejects_template_without_placeholder() -> None:
    assert split_template_at_placeholder("no slot here") is None
