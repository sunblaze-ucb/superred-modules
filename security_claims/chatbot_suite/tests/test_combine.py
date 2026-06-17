"""Tests for the generic combinator: dedup, per-category cap, manifest."""

from __future__ import annotations

import pytest

from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.goal import Goal

from security_claim_chatbot_suite.combine import (
    category_of,
    combine_claims,
    normalize_goal,
)


class _FakeTask(Task):
    """Minimal task exposing a goal and an optional category attribute."""

    def __init__(self, text: str, category: str | None = None) -> None:
        self._goal = Goal(description=text)
        self._category = category

    @property
    def goal(self) -> Goal:
        return self._goal

    # Picked up by category_of via the "category" attribute name.
    @property
    def category(self) -> str | None:
        return self._category

    async def configure_target(self, target) -> None:  # pragma: no cover - unused
        pass

    async def evaluate(
        self, trajectory, target
    ) -> EvaluationResult:  # pragma: no cover
        return EvaluationResult(success=False, primary_score=Score(value=0.0))


def _claim(*tasks: _FakeTask) -> SecurityClaim:
    return SecurityClaim.from_tasks(list(tasks))


def test_normalize_goal_collapses_case_and_whitespace() -> None:
    assert normalize_goal("  Hello   World  ") == normalize_goal("hello world")
    assert normalize_goal("A\nB") == "a b"


def test_category_of_uses_first_populated_attr() -> None:
    assert category_of(_FakeTask("x", category="violence")) == "category=violence"
    assert category_of(_FakeTask("x")) == "_uncategorized"


def test_dedup_drops_normalized_duplicates_first_seen_wins() -> None:
    a = _claim(_FakeTask("Make a bomb", category="c1"))
    b = _claim(_FakeTask("  make   a BOMB ", category="c2"))  # same normalized text
    manifest: list = []
    combined = combine_claims([("a", a), ("b", b)], dedup=True, manifest_out=manifest)
    tasks = list(combined)
    assert len(tasks) == 1
    # first source ("a") wins the duplicate
    assert manifest[0].source == "a"


def test_no_dedup_keeps_duplicates() -> None:
    a = _claim(_FakeTask("dup", category="c1"))
    b = _claim(_FakeTask("dup", category="c2"))
    combined = combine_claims([("a", a), ("b", b)], dedup=False)
    assert len(list(combined)) == 2


def test_per_category_cap() -> None:
    src = _claim(
        _FakeTask("one", category="x"),
        _FakeTask("two", category="x"),
        _FakeTask("three", category="x"),
        _FakeTask("four", category="y"),
    )
    combined = combine_claims([("s", src)], max_per_category=1)
    tasks = list(combined)
    # 1 from category x, 1 from category y
    assert len(tasks) == 2


def test_cap_is_namespaced_across_sources() -> None:
    # Two sources, each one task, both "x" but via different attribute would
    # still collide here since both use .category -> namespaced as category=x.
    a = _claim(_FakeTask("a", category="x"))
    b = _claim(_FakeTask("b", category="x"))
    combined = combine_claims([("a", a), ("b", b)], max_per_category=1)
    assert len(list(combined)) == 1


def test_manifest_indices_are_1_based_and_ordered() -> None:
    src = _claim(_FakeTask("a", category="c1"), _FakeTask("b", category="c2"))
    manifest: list = []
    combine_claims([("s", src)], manifest_out=manifest)
    assert [r.index for r in manifest] == [1, 2]
    assert [r.goal_preview for r in manifest] == ["a", "b"]
    assert all(r.task_class == "_FakeTask" for r in manifest)


def test_empty_after_dedup_raises() -> None:
    # Both tasks are the same normalized text; only one survives -> non-empty.
    # An empty source list must raise.
    with pytest.raises(ValueError):
        combine_claims([])


def test_stats_out_captures_counts() -> None:
    src = _claim(
        _FakeTask("a", category="c1"),
        _FakeTask("a", category="c1"),  # dup
        _FakeTask("b", category="c1"),  # over cap if cap=1
    )
    stats_out: list = []
    combine_claims([("s", src)], max_per_category=1, stats_out=stats_out)
    stats = stats_out[0]
    assert stats.total_input == 3
    assert stats.kept == 1
    assert stats.dropped_duplicate == 1
    assert stats.dropped_over_cap == 1
