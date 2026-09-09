"""Tests for the vendored JBB-Behaviors loader."""

from __future__ import annotations

import csv
from importlib import resources

import pytest
from jailbreakbench_claim.loader import EXPECTED_COLUMNS, all_categories, load_behaviors


def test_loads_all_hundred_behaviours() -> None:
    rows = load_behaviors()
    assert len(rows) == 100


def test_columns_match_upstream_schema() -> None:
    rows = load_behaviors()
    assert tuple(rows[0].keys()) == EXPECTED_COLUMNS


def test_rows_are_byte_identical_to_vendored_csv() -> None:
    """The loader must not reshape upstream rows."""
    text = (
        resources.files("jailbreakbench_claim.data")
        .joinpath("harmful-behaviors.csv")
        .read_text(encoding="utf-8")
    )
    expected = list(csv.DictReader(text.splitlines()))
    assert load_behaviors() == expected


def test_every_row_has_a_goal() -> None:
    assert all(r["Goal"].strip() for r in load_behaviors())


def test_indices_are_zero_to_ninety_nine() -> None:
    rows = load_behaviors()
    assert [r["Index"] for r in rows] == [str(i) for i in range(100)]


def test_filter_by_category() -> None:
    cats = all_categories()
    one = cats[0]
    rows = load_behaviors(categories=[one])
    assert rows
    assert all(r["Category"] == one for r in rows)


def test_filter_by_category_is_case_insensitive() -> None:
    one = all_categories()[0]
    assert load_behaviors(categories=[one.upper()]) == load_behaviors(categories=[one])


def test_filter_by_source() -> None:
    rows = load_behaviors(sources=["Original"])
    assert rows
    assert all(r["Source"] == "Original" for r in rows)


def test_limit_caps_rows_in_order() -> None:
    rows = load_behaviors(limit=5)
    assert len(rows) == 5
    assert rows == load_behaviors()[:5]


def test_unmatched_filter_raises() -> None:
    with pytest.raises(ValueError, match="no behaviours matched"):
        load_behaviors(categories=["NotACategory"])


def test_bad_limit_raises() -> None:
    with pytest.raises(ValueError, match="limit must be"):
        load_behaviors(limit=0)


def test_ten_categories_present() -> None:
    # JBB-Behaviors spans 10 OpenAI-usage-policy categories.
    assert len(all_categories()) == 10


def test_benign_control_set_loads() -> None:
    """JBB ships a paired benign set; it is a control for over-refusal, not an
    attack set."""
    from jailbreakbench_claim.loader import DATASETS, load_behaviors

    assert DATASETS == ("harmful", "benign")
    benign = load_behaviors(dataset="benign")
    assert len(benign) == 100
    assert tuple(benign[0].keys()) == EXPECTED_COLUMNS


def test_benign_and_harmful_are_different_rows() -> None:
    from jailbreakbench_claim.loader import load_behaviors

    assert {r["Goal"] for r in load_behaviors()} != {
        r["Goal"] for r in load_behaviors(dataset="benign")
    }


def test_unknown_dataset_raises() -> None:
    from jailbreakbench_claim.loader import load_behaviors

    with pytest.raises(ValueError, match="unknown dataset"):
        load_behaviors(dataset="nope")
