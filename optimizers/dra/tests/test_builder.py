"""Tests for DRA prompt generation parity helpers."""

from __future__ import annotations

import re
from random import Random

from dra_optimizer.builder import DraPromptBuilder, OFFICIAL_SYSTEM_PROMPT, normalize_model_name


def test_prefix_gen_recovers_lowercase_question_from_parenthesized_letters() -> None:
    builder = DraPromptBuilder(rng=Random(7))

    prefix, current_question = builder.prefix_gen("Ab C!")

    recovered = "".join(re.findall(r"\((.)\)", prefix))
    assert current_question == "ab c!"
    assert recovered == current_question


def test_build_uses_official_system_prompt_and_reconstruction_suffix() -> None:
    builder = DraPromptBuilder(rng=Random(3))

    prompt = builder.build(question="alpha beta", model="llama")

    assert prompt.system_prompt == OFFICIAL_SYSTEM_PROMPT
    assert prompt.disguised_question == "alpha beta"
    assert "construct a sentence named Q1" in prompt.user_prompt
    assert "Note that Q1 contains" in prompt.user_prompt
    assert prompt.model == "llama"


def test_model_normalization_maps_visible_model_names_to_official_branches() -> None:
    assert normalize_model_name("gpt-4o-2024-05-13") == "gpt-4o"
    assert normalize_model_name("gpt4") == "gpt-4"
    assert normalize_model_name("Llama-2-13b-chat") == "llama"
    assert normalize_model_name("mini") == "llama"
    assert normalize_model_name(None) == "llama"


def test_sensitive_token_detector_controls_word_guide() -> None:
    builder = DraPromptBuilder(rng=Random(11), sensitive_token_detector=lambda token: token == "illegally")

    prompt = builder.build(question="test illegally", model="gpt-4")

    assert "illegally" not in prompt.word_guide or "`illegally`" not in prompt.word_guide
    assert prompt.word_guide
