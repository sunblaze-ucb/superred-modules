"""Loader for MT-JailBench's vendored HarmBench behavior CSV.

MT-JailBench evaluates over the HarmBench standard behaviors
(``data/harmbench_behaviors.csv``; ``data/harmbench.py::load_datasets`` filters
``FunctionalCategory == "standard"``). We vendor that CSV byte-for-byte and read
it with the stdlib :mod:`csv` module (upstream uses pandas + numpy, which this
port drops).

The behavior strings themselves are HarmBench content, redistributed by
MT-JailBench under the MIT License; HarmBench is attributed separately in
``LICENSES/``. This loader returns only structure (rows / counts), never the
bodies to any log.
"""

from __future__ import annotations

import csv
from importlib.resources import as_file, files
from typing import Final, TypedDict

_DATA_PACKAGE: Final = "mt_jailbench_claim.data"
_CSV: Final = "harmbench_behaviors.csv"

#: Verbatim upstream columns.
EXPECTED_COLUMNS: Final = frozenset(
    {"Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID"}
)

#: The functional category MT-JailBench selects (``load_datasets`` default).
STANDARD_CATEGORY: Final = "standard"


class BehaviorRow(TypedDict):
    """One HarmBench behavior row."""

    Behavior: str
    FunctionalCategory: str
    SemanticCategory: str
    Tags: str
    ContextString: str
    BehaviorID: str


def _read_rows() -> list[BehaviorRow]:
    resource = files(_DATA_PACKAGE).joinpath(_CSV)
    with as_file(resource) as path, open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("empty or unreadable behavior CSV")
        missing = EXPECTED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"behavior CSV missing required columns: {sorted(missing)!r}")
        return [BehaviorRow({col: row[col] for col in EXPECTED_COLUMNS}) for row in reader]  # type: ignore[misc]


def load_standard_behaviors(*, number_of_behaviors: int | None = None) -> list[BehaviorRow]:
    """Return the standard HarmBench behaviors (upstream's task set).

    Args:
        number_of_behaviors: If given, keep only the first N (upstream's
            ``num_behaviors``); ``None`` (shipped default) keeps all 159.

    Returns:
        Rows in CSV order (deterministic). 159 rows when unfiltered.
    """
    rows = [r for r in _read_rows() if r["FunctionalCategory"] == STANDARD_CATEGORY]
    if number_of_behaviors is not None:
        rows = rows[:number_of_behaviors]
    if not rows:
        raise ValueError("no standard behaviors selected")
    return rows
