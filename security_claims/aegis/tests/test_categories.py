"""The taxonomy constants must match the labels in the vendored test split."""

from __future__ import annotations

from aegis_claim.categories import (
    ALL_LABELS,
    HARM_CATEGORIES,
    NEEDS_CAUTION,
    SAFE,
    atomic_categories,
    is_unsafe,
    slugify,
)
from aegis_claim.loader import load_all_rows


def test_thirteen_harm_categories() -> None:
    assert len(HARM_CATEGORIES) == 13
    assert len(set(HARM_CATEGORIES)) == 13


def test_harm_categories_match_dataset_labels() -> None:
    """Every atomic label in the split is a known harm category, Needs Caution, or Safe."""
    seen: set[str] = set()
    for row in load_all_rows():
        seen.update(atomic_categories(row["labels_0"]))
    assert seen == set(ALL_LABELS), (
        f"category vocabulary drifted: only-in-data={seen - set(ALL_LABELS)}, "
        f"only-in-module={set(ALL_LABELS) - seen}"
    )


def test_all_labels_is_harm_plus_needs_caution_plus_safe() -> None:
    assert set(ALL_LABELS) == {*HARM_CATEGORIES, NEEDS_CAUTION, SAFE}
    assert NEEDS_CAUTION not in HARM_CATEGORIES
    assert SAFE not in HARM_CATEGORIES


def test_atomic_categories_splits_multilabel() -> None:
    assert atomic_categories("Criminal Planning/Confessions, Violence") == [
        "Criminal Planning/Confessions",
        "Violence",
    ]
    assert atomic_categories("Violence") == ["Violence"]


def test_is_unsafe_semantics() -> None:
    assert is_unsafe("Violence") is True
    assert is_unsafe("Criminal Planning/Confessions, Violence") is True
    assert is_unsafe("Safe") is False
    # Needs Caution: included by default, excluded when asked.
    assert is_unsafe("Needs Caution") is True
    assert is_unsafe("Needs Caution", include_needs_caution=False) is False
    # A harm category is unsafe even with include_needs_caution=False.
    assert is_unsafe("Violence", include_needs_caution=False) is True


def test_slugify() -> None:
    assert slugify("Hate/Identity Hate") == "hate_identity_hate"
    assert slugify("Sexual (minor)") == "sexual_minor"
    assert slugify("PII/Privacy") == "pii_privacy"
