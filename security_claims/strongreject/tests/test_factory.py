"""Public factory functions: counts, ordering, statelessness."""

from __future__ import annotations

from chatbot_target import ChatbotTarget
from superred.core.interfaces.security_claim import SecurityClaim

from strongreject_claim import (
    strongreject_disinformation_claim,
    strongreject_full_claim,
    strongreject_hate_claim,
    strongreject_illegal_goods_claim,
    strongreject_nonviolent_crimes_claim,
    strongreject_sexual_content_claim,
    strongreject_small_claim,
    strongreject_violence_claim,
)
from strongreject_claim.loader import load_full_rows, load_small_rows
from strongreject_claim.task import StrongREJECTTask

_ARGS = {"judge_api_base": "x", "judge_api_key": "y"}


def _count(claim: SecurityClaim[ChatbotTarget]) -> int:
    return sum(1 for _ in claim)


def test_violence_claim_iterates_54_tasks() -> None:
    assert _count(strongreject_violence_claim(**_ARGS)) == 54


def test_disinformation_claim_iterates_50_tasks() -> None:
    assert _count(strongreject_disinformation_claim(**_ARGS)) == 50


def test_hate_claim_iterates_50_tasks() -> None:
    assert _count(strongreject_hate_claim(**_ARGS)) == 50


def test_illegal_goods_claim_iterates_50_tasks() -> None:
    assert _count(strongreject_illegal_goods_claim(**_ARGS)) == 50


def test_nonviolent_crimes_claim_iterates_59_tasks() -> None:
    assert _count(strongreject_nonviolent_crimes_claim(**_ARGS)) == 59


def test_sexual_content_claim_iterates_50_tasks() -> None:
    assert _count(strongreject_sexual_content_claim(**_ARGS)) == 50


def test_full_claim_iterates_313_tasks() -> None:
    assert _count(strongreject_full_claim(**_ARGS)) == 313


def test_full_claim_iteration_order_groups_by_category() -> None:
    """Iteration follows the documented order:
    disinformation, hate, illegal_goods, nonviolent_crimes,
    sexual_content, violence.
    """
    expected_order = [
        ("disinformation and deception", 50),
        ("hate, harassment and discrimination", 50),
        ("illegal goods and services", 50),
        ("non-violent crimes", 59),
        ("sexual content", 50),
        ("violence", 54),
    ]
    tasks = list(strongreject_full_claim(**_ARGS))
    cursor = 0
    for category, count in expected_order:
        block = tasks[cursor : cursor + count]
        assert all(isinstance(t, StrongREJECTTask) for t in block)
        assert all(t.category == category for t in block)  # type: ignore[attr-defined]
        cursor += count
    assert cursor == 313


def test_full_claim_is_reiterable() -> None:
    """Tasks are stateless; iterating twice yields the same number of tasks."""
    claim = strongreject_full_claim(**_ARGS)
    first = list(claim)
    second = list(claim)
    assert len(first) == len(second) == 313


def test_small_claim_iterates_60_tasks() -> None:
    assert _count(strongreject_small_claim(**_ARGS)) == 60


def test_small_categories_normalised_to_lowercase() -> None:
    tasks = list(strongreject_small_claim(**_ARGS))
    categories = {t.category for t in tasks}  # type: ignore[attr-defined]
    assert categories == {
        "disinformation and deception",
        "hate, harassment and discrimination",
        "illegal goods and services",
        "non-violent crimes",
        "sexual content",
        "violence",
    }


def test_small_prompts_are_strict_subset_of_full() -> None:
    """Every small-CSV forbidden_prompt also appears in the full CSV."""
    full_prompts = {r["forbidden_prompt"] for r in load_full_rows()}
    small_prompts = {r["forbidden_prompt"] for r in load_small_rows()}
    assert small_prompts.issubset(full_prompts)
    assert len(small_prompts) == 60


def test_factories_propagate_judge_config_to_tasks() -> None:
    """All tasks share the JudgeConfig built from factory args."""
    claim = strongreject_violence_claim(
        judge_api_base="https://example.invalid/v1",
        judge_api_key="secret-not-real",
        judge_max_retries=7,
    )
    for task in claim:
        assert isinstance(task, StrongREJECTTask)
        assert task._judge_config.api_base == "https://example.invalid/v1"  # type: ignore[attr-defined]
        assert task._judge_config.api_key == "secret-not-real"  # type: ignore[attr-defined]
        assert task._judge_config.max_retries == 7  # type: ignore[attr-defined]


def test_factories_propagate_success_threshold_to_tasks() -> None:
    claim = strongreject_violence_claim(success_threshold=0.5, **_ARGS)
    for task in claim:
        assert isinstance(task, StrongREJECTTask)
        assert task._success_threshold == 0.5  # type: ignore[attr-defined]
