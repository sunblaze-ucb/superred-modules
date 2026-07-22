"""Category discovery + validation against the real dataset (cached, offline)."""

from __future__ import annotations

import pytest

from agentharm_claim.categories import (
    EXPECTED_CATEGORIES,
    category_slug,
    discover_categories,
    validate_categories,
)
from agentharm_claim.dataset_loader import load_agentharm_dataset


def test_expected_categories_count() -> None:
    assert len(EXPECTED_CATEGORIES) == 8
    assert "Fraud" in EXPECTED_CATEGORIES and "Sexual" in EXPECTED_CATEGORIES


def test_discover_and_validate_against_real_dataset() -> None:
    ds = load_agentharm_dataset()
    assert discover_categories(ds) == EXPECTED_CATEGORIES
    assert validate_categories(ds) == EXPECTED_CATEGORIES


def test_validate_raises_on_drift() -> None:
    class _S:
        metadata = {"category": "NotARealCategory"}

    class _DS:
        def __iter__(self):
            return iter([_S()])

    with pytest.raises(ValueError, match="drifted"):
        validate_categories(_DS())  # type: ignore[arg-type]


def test_category_slug() -> None:
    assert category_slug("Hate") == "hate"
    assert category_slug("Disinformation") == "disinformation"
