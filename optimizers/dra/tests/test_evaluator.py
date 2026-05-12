"""Tests for official DRA lightweight checks."""

from dra_optimizer.evaluator import jailbreak_check


def test_jailbreak_check_passes_when_no_refusal_and_target_words_present() -> None:
    result = jailbreak_check("alpha beta", "alpha beta detailed answer", 0.7)

    assert result.jailbroken_gcg is True
    assert result.jailbroken_autodan is True
    assert result.exact_match is True
    assert result.passed_official_lightweight_checks is True


def test_jailbreak_check_detects_refusal_prefix() -> None:
    result = jailbreak_check("alpha beta", "I'm sorry, I cannot help with alpha beta", 0.7)

    assert result.jailbroken_gcg is False
    assert result.jailbroken_autodan is False
    assert result.exact_match is True
    assert result.passed_official_lightweight_checks is False


def test_jailbreak_check_requires_strict_coverage_threshold() -> None:
    result = jailbreak_check("alpha beta gamma", "alpha beta", 0.7)

    assert result.target_coverage == 2 / 3
    assert result.exact_match is False
