"""Factory tests: cardinalities, type, composition, attribute exposure."""

from __future__ import annotations

from harmbench_claim import (
    harmbench_claim,
    harmbench_complete,
    harmbench_contextual_test,
    harmbench_contextual_val,
    harmbench_copyright_test,
    harmbench_copyright_val,
    harmbench_standard_test,
    harmbench_standard_val,
    harmbench_test,
    harmbench_val,
)


_JUDGE_KWARGS = {
    "judge_model": "openai/test",
    "judge_api_base": "https://x",
    "judge_api_key": "sk-test",
}


# --- Leaf factory cardinalities (the dataset values) ---

LEAF_EXPECTED = {
    harmbench_standard_test: 159,
    harmbench_contextual_test: 81,
    harmbench_copyright_test: 80,
    harmbench_standard_val: 41,
    harmbench_contextual_val: 19,
    harmbench_copyright_val: 20,
}


def test_leaf_factory_cardinalities() -> None:
    for factory, expected in LEAF_EXPECTED.items():
        tasks = list(factory(**_JUDGE_KWARGS))
        assert len(tasks) == expected, (
            f"{factory.__name__}: expected {expected} tasks, got {len(tasks)}"
        )


def test_test_split_total() -> None:
    tasks = list(harmbench_test(**_JUDGE_KWARGS))
    assert len(tasks) == 159 + 81 + 80  # 320


def test_val_split_total() -> None:
    tasks = list(harmbench_val(**_JUDGE_KWARGS))
    assert len(tasks) == 41 + 19 + 20  # 80


def test_complete_total() -> None:
    tasks = list(harmbench_complete(**_JUDGE_KWARGS))
    assert len(tasks) == 320 + 80  # 400


def test_complete_re_iterable() -> None:
    """SecurityClaim.from_claims is lazy; it must support re-iteration
    because tasks are stateless."""
    claim = harmbench_complete(**_JUDGE_KWARGS)
    first = list(claim)
    second = list(claim)
    assert len(first) == len(second)
    assert [t.behavior_id for t in first] == [t.behavior_id for t in second]  # type: ignore[attr-defined]


def test_task_attributes_propagate() -> None:
    tasks = list(harmbench_standard_test(**_JUDGE_KWARGS))
    for task in tasks:
        assert task.functional_category == "standard"  # type: ignore[attr-defined]
        assert task.split == "test"  # type: ignore[attr-defined]
        assert task.behavior_id  # type: ignore[attr-defined]


def test_factory_default_judge_model() -> None:
    """When no judge_model is given, default is openai/gpt-4o-mini."""
    # Need to provide api creds even for default model; use stubs.
    tasks = list(harmbench_standard_val(
        judge_api_base="https://x", judge_api_key="sk-test",
    ))
    # The judge model on the underlying task should be the default.
    assert tasks[0].judge.model == "openai/gpt-4o-mini"  # type: ignore[attr-defined]


def test_parameterized_factory_csv_path_override(tmp_path) -> None:
    """csv_path lets users ship their own behavior set without rebuilding."""
    custom_csv = tmp_path / "custom.csv"
    custom_csv.write_text(
        "Behavior,FunctionalCategory,SemanticCategory,Tags,ContextString,BehaviorID\n"
        "test behavior,standard,harmful,,,custom1\n"
        "another behavior,standard,illegal,,,custom2\n",
        encoding="utf-8",
    )
    tasks = list(harmbench_claim(
        split="test", csv_path=str(custom_csv), **_JUDGE_KWARGS,
    ))
    assert len(tasks) == 2
    assert tasks[0].behavior_id == "custom1"  # type: ignore[attr-defined]
    assert tasks[1].behavior_id == "custom2"  # type: ignore[attr-defined]
    assert tasks[0].split == "custom"  # type: ignore[attr-defined]


def test_semantic_categories_filter_works() -> None:
    tasks = list(harmbench_claim(
        split="test",
        functional_categories=("standard",),
        semantic_categories=("chemical_biological",),
        **_JUDGE_KWARGS,
    ))
    # 42 chemical_biological standard rows in test split (verified via the CSV).
    assert len(tasks) > 0
    for task in tasks:
        assert task.semantic_category == "chemical_biological"  # type: ignore[attr-defined]
        assert task.functional_category == "standard"  # type: ignore[attr-defined]
