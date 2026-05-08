"""Loader tests: exact row counts, schema, filters, deterministic ordering."""

from __future__ import annotations

import pytest

from secclaim_harmbench.loader import VALID_FUNCTIONAL, VALID_SPLITS, load_behaviors


# Verified counts from the bundled CSVs (see PROVENANCE.md):
#   test: 320 = 159 standard + 81 contextual + 80 copyright
#   val:   80 = 41 standard + 19 contextual + 20 copyright

EXPECTED_TEST_TOTAL = 320
EXPECTED_VAL_TOTAL = 80
EXPECTED_PER_SPLIT_FUNCTIONAL = {
    ("test", "standard"): 159,
    ("test", "contextual"): 81,
    ("test", "copyright"): 80,
    ("val", "standard"): 41,
    ("val", "contextual"): 19,
    ("val", "copyright"): 20,
}


def test_test_split_total_count() -> None:
    rows = load_behaviors(split="test")
    assert len(rows) == EXPECTED_TEST_TOTAL


def test_val_split_total_count() -> None:
    rows = load_behaviors(split="val")
    assert len(rows) == EXPECTED_VAL_TOTAL


@pytest.mark.parametrize(
    ("split", "functional", "expected"),
    [(s, f, c) for (s, f), c in EXPECTED_PER_SPLIT_FUNCTIONAL.items()],
)
def test_per_split_functional_counts(split: str, functional: str, expected: int) -> None:
    rows = load_behaviors(split=split, functional_categories=(functional,))  # type: ignore[arg-type]
    assert len(rows) == expected
    for row in rows:
        assert row["FunctionalCategory"] == functional
        assert row["_split"] == split


def test_schema_has_expected_columns() -> None:
    rows = load_behaviors(split="val", functional_categories=("standard",))
    required = {
        "Behavior", "FunctionalCategory", "SemanticCategory", "Tags",
        "ContextString", "BehaviorID", "_split",
    }
    assert required <= set(rows[0].keys())


def test_contextual_rows_have_non_empty_context() -> None:
    rows = load_behaviors(split="test", functional_categories=("contextual",))
    for row in rows:
        assert row["ContextString"], (
            f"Contextual row {row['BehaviorID']!r} has empty ContextString"
        )


def test_copyright_rows_have_hash_check_tag() -> None:
    rows = load_behaviors(split="test", functional_categories=("copyright",))
    for row in rows:
        tags = {t.strip() for t in row["Tags"].split(",") if t.strip()}
        assert "hash_check" in tags, (
            f"Copyright row {row['BehaviorID']!r} missing hash_check tag"
        )
        assert tags & {"book", "lyrics"}, (
            f"Copyright row {row['BehaviorID']!r} missing book/lyrics tag"
        )


def test_standard_rows_have_no_tags() -> None:
    rows = load_behaviors(split="test", functional_categories=("standard",))
    for row in rows:
        # Standard rows have empty Tags string upstream.
        assert row["Tags"] == "", (
            f"Standard row {row['BehaviorID']!r} has unexpected Tags={row['Tags']!r}"
        )


def test_invalid_split_raises() -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        load_behaviors(split="train")  # type: ignore[arg-type]


def test_invalid_functional_category_raises() -> None:
    with pytest.raises(ValueError, match="unknown functional_categories"):
        load_behaviors(split="val", functional_categories=("not_a_real_category",))


def test_filters_zero_rows_raises() -> None:
    with pytest.raises(ValueError, match="filters selected zero rows"):
        load_behaviors(
            split="val",
            functional_categories=("standard",),
            semantic_categories=("definitely_not_a_real_category",),
        )


def test_semantic_filter_works() -> None:
    rows = load_behaviors(split="test", semantic_categories=("copyright",))
    # All copyright SemanticCategory rows are also FunctionalCategory=copyright.
    assert len(rows) == 80
    for row in rows:
        assert row["SemanticCategory"] == "copyright"


def test_deterministic_ordering() -> None:
    """Two loads produce identical row order."""
    rows1 = load_behaviors(split="val")
    rows2 = load_behaviors(split="val")
    ids1 = [r["BehaviorID"] for r in rows1]
    ids2 = [r["BehaviorID"] for r in rows2]
    assert ids1 == ids2


def test_test_val_disjoint() -> None:
    """No BehaviorID appears in both splits (verified upstream too)."""
    test_ids = {r["BehaviorID"] for r in load_behaviors(split="test")}
    val_ids = {r["BehaviorID"] for r in load_behaviors(split="val")}
    assert test_ids & val_ids == set()


def test_valid_functional_constant() -> None:
    assert VALID_FUNCTIONAL == frozenset({"standard", "contextual", "copyright"})


def test_valid_splits_constant() -> None:
    assert set(VALID_SPLITS) == {"test", "val"}
