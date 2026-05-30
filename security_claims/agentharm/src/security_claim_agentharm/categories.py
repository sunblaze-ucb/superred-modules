"""Harm categories present in the AgentHarm harmful test_public split.

The full AgentHarm paper defines 11 harm categories, but the *released* public
test split (44 base behaviors x 4 augmentations = 176) covers 8 of them; the
remaining categories live in the withheld / private splits.  We therefore
subdivide the 176-case claim into these 8 per-category subclaims.

``EXPECTED_CATEGORIES`` is pinned from a real dataset load; ``validate_categories``
fails loudly if a future dataset revision drifts from it.
"""

from __future__ import annotations

from inspect_ai.dataset import Dataset

EXPECTED_CATEGORIES: frozenset[str] = frozenset(
    {
        "Copyright",
        "Cybercrime",
        "Disinformation",
        "Drugs",
        "Fraud",
        "Harassment",
        "Hate",
        "Sexual",
    }
)
"""The 8 harm categories in the harmful test_public split (pinned from a real load)."""


def discover_categories(ds: Dataset) -> frozenset[str]:
    """Collect the distinct ``metadata['category']`` values in *ds*."""
    out: set[str] = set()
    for sample in ds:
        if sample.metadata is not None:
            out.add(str(sample.metadata["category"]))
    return frozenset(out)


def validate_categories(ds: Dataset) -> frozenset[str]:
    """Return the dataset's categories, raising if they drift from EXPECTED_CATEGORIES."""
    found = discover_categories(ds)
    if found != EXPECTED_CATEGORIES:
        missing = EXPECTED_CATEGORIES - found
        extra = found - EXPECTED_CATEGORIES
        raise ValueError(
            "AgentHarm categories drifted from EXPECTED_CATEGORIES "
            f"(missing={sorted(missing)}, extra={sorted(extra)}). "
            "The dataset revision may have changed; update EXPECTED_CATEGORIES."
        )
    return found


def category_slug(category: str) -> str:
    """A lowercase identifier-safe slug for a category (e.g. 'Hate' -> 'hate')."""
    return category.strip().lower().replace(" ", "_").replace("-", "_")


__all__ = [
    "EXPECTED_CATEGORIES",
    "discover_categories",
    "validate_categories",
    "category_slug",
]
