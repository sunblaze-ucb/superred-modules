"""Loaders for the bundled StrongREJECT CSVs and rubric judge prompts.

The CSVs and prompt templates are vendored under ``data/`` and ``prompts/``
beside this module so that runs are reproducible against the exact files
the implementation plan was reviewed against (sha256-pinned in the
unit-test suite). Loading goes through :func:`importlib.resources.files`
so the package is installable as a wheel without a source tree.

Two upstream encoding facts that influenced this loader:

1. The full CSV contains one prompt with non-ASCII characters and two
   prompts with embedded newlines. We open with ``encoding="utf-8"`` and
   ``newline=""`` so that :class:`csv.DictReader` round-trips both
   correctly.
2. The full CSV uses category label ``"Disinformation and deception"``
   while the small CSV uses ``"Disinformation and Deception"``. We
   normalise the category column to lowercase at load time so a single
   key set works across both files.
"""

from __future__ import annotations

import csv
from importlib.resources import as_file, files
from typing import Final, TypedDict


class StrongREJECTRow(TypedDict):
    """One row from the bundled StrongREJECT CSV.

    Attributes:
        category: Lowercased category label (canonical 6 categories).
        source: Origin dataset name (``custom``, ``DAN``, ``AdvBench``, etc.).
        forbidden_prompt: The forbidden prompt verbatim from upstream.
    """

    category: str
    source: str
    forbidden_prompt: str


_DATA_PACKAGE: Final = "strongreject_claim.data"
_FULL_CSV: Final = "strongreject_dataset.csv"
_SMALL_CSV: Final = "strongreject_small_dataset.csv"


def _read_csv(filename: str) -> list[StrongREJECTRow]:
    """Read a bundled CSV and return its rows with category lowercased.

    Embedded newlines inside quoted prompt cells require ``newline=""``
    to be passed through to the underlying file object, otherwise the
    CSV reader treats them as record terminators. ``importlib.resources``
    Traversables do not expose ``newline``, so we materialise to a real
    filesystem path via :func:`importlib.resources.as_file` (which is a
    no-op when the resource is already a file on disk, as it is for an
    editable install) and use the standard :func:`open` builtin.
    """
    csv_resource = files(_DATA_PACKAGE).joinpath(filename)
    with as_file(csv_resource) as path, open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[StrongREJECTRow] = []
        for raw in reader:
            rows.append(
                StrongREJECTRow(
                    category=raw["category"].lower(),
                    source=raw["source"],
                    forbidden_prompt=raw["forbidden_prompt"],
                )
            )
        return rows


def load_full_rows() -> list[StrongREJECTRow]:
    """Return all 313 rows from the canonical full StrongREJECT CSV."""
    return _read_csv(_FULL_CSV)


def load_small_rows() -> list[StrongREJECTRow]:
    """Return all 60 rows from the curated StrongREJECT-small CSV."""
    return _read_csv(_SMALL_CSV)


def rows_for_category(category: str) -> list[StrongREJECTRow]:
    """Return the rows in the full dataset whose lowercased category matches.

    The match is exact on the lowercased category string. Unknown
    categories return an empty list rather than raising, so factory
    functions can be composed without knowing whether a category exists.
    """
    needle = category.lower()
    return [r for r in load_full_rows() if r["category"] == needle]
