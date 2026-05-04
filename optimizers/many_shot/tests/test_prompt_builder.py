"""Tests for many-shot prompt assembly."""

from many_shot_optimizer.prompt_builder import build_prompt


def test_build_prompt_basic():
    examples = [
        {"user": "Q1", "assistant": "A1"},
        {"user": "Q2", "assistant": "A2"},
    ]
    result = build_prompt(examples=examples, objective="Do something bad")
    assert result.startswith("You are a helpful assistant")
    assert "User: Q1\nAssistant: A1" in result
    assert "User: Q2\nAssistant: A2" in result
    assert result.endswith("User: Do something bad")


def test_build_prompt_preserves_order():
    examples = [
        {"user": f"Q{i}", "assistant": f"A{i}"}
        for i in range(5)
    ]
    result = build_prompt(examples=examples, objective="final")
    positions = [result.index(f"Q{i}") for i in range(5)]
    assert positions == sorted(positions)


def test_build_prompt_empty_examples():
    result = build_prompt(examples=[], objective="test")
    assert "You are a helpful assistant" in result
    assert result.endswith("User: test")


def test_build_prompt_example_count():
    examples = [{"user": f"Q{i}", "assistant": f"A{i}"} for i in range(10)]
    result = build_prompt(examples=examples, objective="X")
    assert result.count("Assistant:") == 10
