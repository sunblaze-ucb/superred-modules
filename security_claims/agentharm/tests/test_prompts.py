"""Faithful prompt construction matches upstream (compared, not hardcoded)."""

from __future__ import annotations

from inspect_evals.agentharm.prompts import get_jailbreak_template, get_system_prompt

from agentharm_claim.prompts import faithful_system_prompt, faithful_user_prompt


def test_system_prompt_default_model() -> None:
    assert faithful_system_prompt("openai/gpt-4o-mini") == get_system_prompt("default", "standard")


def test_system_prompt_llama_model() -> None:
    assert faithful_system_prompt("meta/llama-3.1-8b-instruct") == get_system_prompt(
        "llama-3.1", "standard"
    )


def test_user_prompt_matches_empty_template() -> None:
    up = faithful_user_prompt("hello world")
    assert up == get_jailbreak_template("empty").format(prompt="hello world")
    assert up.startswith("\n\n")
    assert up.endswith("hello world")
