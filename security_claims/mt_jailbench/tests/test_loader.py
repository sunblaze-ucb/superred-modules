"""Loader tests: the standard subset is 159 rows, read with the stdlib csv."""

from __future__ import annotations

import csv
import sys
from importlib.resources import as_file, files

import pytest

from mt_jailbench_claim.loader import (
    EXPECTED_COLUMNS,
    STANDARD_CATEGORY,
    load_standard_behaviors,
)


def test_standard_subset_is_159() -> None:
    rows = load_standard_behaviors()
    assert len(rows) == 159
    assert all(r["FunctionalCategory"] == STANDARD_CATEGORY for r in rows)


def test_rows_have_expected_columns() -> None:
    row = load_standard_behaviors()[0]
    assert EXPECTED_COLUMNS <= set(row.keys())


def test_number_of_behaviors_truncates_in_order() -> None:
    head = load_standard_behaviors(number_of_behaviors=5)
    assert len(head) == 5
    assert head == load_standard_behaviors()[:5]


def test_full_csv_has_320_rows_and_no_pandas() -> None:
    resource = files("mt_jailbench_claim.data").joinpath("harmbench_behaviors.csv")
    with as_file(resource) as path, open(path, encoding="utf-8", newline="") as fh:
        total = sum(1 for _ in csv.DictReader(fh))
    assert total == 320
    # the loader path must not have imported pandas/numpy
    assert "pandas" not in sys.modules


def test_loader_is_deterministic() -> None:
    assert load_standard_behaviors() == load_standard_behaviors()


def test_bad_number_raises_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        load_standard_behaviors(number_of_behaviors=0)
