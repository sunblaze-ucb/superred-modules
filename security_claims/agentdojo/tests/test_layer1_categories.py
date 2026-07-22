"""Tests for the Layer-1 category taxonomy.

Verifies MECE: every canonical pair maps to exactly one category, and
every (suite, injection_task) pair the user confirmed is present.
"""

from __future__ import annotations

from agentdojo_claim.layer1_categories import (
    ALL_CATEGORIES,
    CATEGORIES_BY_SUITE,
    INJECTION_CATEGORIES,
    category_of,
)
from agentdojo_claim.layer1_pairs import CANONICAL_PAIRS


def test_every_canonical_pair_has_a_category() -> None:
    """No canonical pair is left without a category label."""
    for suite, _ut, it in CANONICAL_PAIRS:
        assert (suite, it) in INJECTION_CATEGORIES, (suite, it)


def test_categories_by_suite_consistent_with_map() -> None:
    """CATEGORIES_BY_SUITE matches the per-suite categories in INJECTION_CATEGORIES."""
    derived: dict[str, set[str]] = {}
    for (suite, _it), cat in INJECTION_CATEGORIES.items():
        derived.setdefault(suite, set()).add(cat)
    for suite, expected in CATEGORIES_BY_SUITE.items():
        assert derived[suite] == set(expected), suite


def test_all_categories_union_complete() -> None:
    """ALL_CATEGORIES = union of per-suite categories."""
    union: set[str] = set()
    for cats in CATEGORIES_BY_SUITE.values():
        union |= set(cats)
    assert ALL_CATEGORIES == union


def test_27_pairs_27_category_entries() -> None:
    assert len(CANONICAL_PAIRS) == 27
    assert len(INJECTION_CATEGORIES) == 27


def test_category_of_helper() -> None:
    assert category_of("banking", "injection_task_0") == "exfil_via_memo"
    assert category_of("workspace", "injection_task_2") == "calendar_manip"
    assert category_of("travel", "injection_task_6") == "output_manip"


def test_calendar_manip_label_shared_across_suites() -> None:
    """workspace IT2 and travel IT2 both use 'calendar_manip' label."""
    assert category_of("workspace", "injection_task_2") == "calendar_manip"
    assert category_of("travel", "injection_task_2") == "calendar_manip"
