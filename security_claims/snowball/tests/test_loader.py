"""Tests for the snowball question loader."""

from __future__ import annotations

import json
from importlib import resources

import pytest
from snowball_claim.loader import SUBSET_NAMES, load_items


def test_subset_names() -> None:
    assert SUBSET_NAMES == ("graph_connectivity", "primes", "senators")


def test_loader_default_is_full_500_per_subset() -> None:
    # The low-level loader defaults to all questions; the factory applies the
    # 100/subset cap (see test_task). 3 subsets x 500 = 1500.
    items = load_items()
    assert len(items) == 1500
    counts = {s: sum(1 for i in items if i.subset == s) for s in SUBSET_NAMES}
    assert counts == {"graph_connectivity": 500, "primes": 500, "senators": 500}


def test_explicit_limit_caps_each_subset() -> None:
    items = load_items(limit=100)
    assert len(items) == 300


def test_limit_none_loads_full_500_each() -> None:
    items = load_items(subsets=["primes"], limit=None)
    assert len(items) == 500


def test_polarity_is_fixed_per_subset() -> None:
    items = load_items()
    pol = {i.subset: i.polarity for i in items}
    assert pol == {
        "graph_connectivity": "yes",
        "primes": "no",
        "senators": "yes",
    }


def test_primes_questions_are_extracted_from_dicts() -> None:
    items = load_items(subsets=["primes"], limit=3)
    assert all("prime" in i.question.lower() for i in items)


def test_limit_takes_from_file_end_like_garak() -> None:
    """garak uses self.prompts[-limit:]; we must match that slice."""
    raw = json.loads(
        resources.files("snowball_claim.data")
        .joinpath("graph_connectivity.json")
        .read_text("utf-8")
    )
    items = load_items(subsets=["graph_connectivity"], limit=5)
    assert [i.question for i in items] == [str(x) for x in raw[-5:]]


def test_subset_order_preserved() -> None:
    items = load_items(subsets=["senators", "primes"], limit=2)
    assert [i.subset for i in items] == ["senators", "senators", "primes", "primes"]


def test_unknown_subset_raises() -> None:
    with pytest.raises(ValueError, match="unknown subset"):
        load_items(subsets=["nope"])


def test_bad_limit_raises() -> None:
    with pytest.raises(ValueError, match="limit must be"):
        load_items(limit=0)
