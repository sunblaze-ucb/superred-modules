"""Loader for the vendored Aegis 1.0 test split.

The dataset ships as a parquet on HuggingFace; we vendor it as both the
byte-identical parquet (provenance anchor, see ``scripts/sync_upstream.py``) and
a deterministic CSV rendering. This module reads the **CSV** with the stdlib
:mod:`csv` module — no pandas, no pyarrow at runtime.

Loading goes through :func:`importlib.resources.files` so the package works
installed as a wheel. The ``text`` column contains embedded newlines, commas and
quotes, so the file is opened with ``newline=""`` (via
:func:`importlib.resources.as_file`) and :class:`csv.DictReader` round-trips
them.

**Output discipline:** this module never logs, prints, or otherwise emits the
``text`` column. Callers receive rows and are responsible for the same
discipline (the task and judge never surface prompt or response text either).
"""

from __future__ import annotations

import csv
from importlib.resources import as_file, files
from typing import Final, TypedDict

from aegis_claim.categories import is_unsafe

_VENDOR_PACKAGE: Final = "aegis_claim._vendor"
_CSV_FILENAME: Final = "aegis_test.csv"

#: text_type of a standalone user prompt (as opposed to an elicited LLM
#: response, a system+user "combined" turn, or a "multi_turn" dialog). A
#: red-team claim sends the user's prompt to the target, so this is the only
#: text_type we use.
USER_MESSAGE_TEXT_TYPE: Final = "user_message"


class AegisRow(TypedDict):
    """One row from the vendored Aegis test split.

    Attributes:
        id: The dataset's row id (a UUID-like string; not a secret).
        text: The prompt text, verbatim. NEVER logged or emitted.
        text_type: One of ``user_message`` / ``combined`` / ``llm_response`` /
            ``multi_turn``.
        labels_0: The aggregated safety label — ``"Safe"``, ``"Needs Caution"``,
            or one or more comma-joined harm categories.
    """

    id: str
    text: str
    text_type: str
    labels_0: str


def _read_all_rows() -> list[AegisRow]:
    """Read every row of the vendored CSV (all text_types, all labels)."""
    resource = files(_VENDOR_PACKAGE).joinpath(_CSV_FILENAME)
    with as_file(resource) as path, open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            AegisRow(
                id=raw["id"],
                text=raw["text"],
                text_type=raw["text_type"],
                labels_0=raw["labels_0"],
            )
            for raw in reader
        ]


def load_all_rows() -> list[AegisRow]:
    """Return all rows of the vendored Aegis test split (1199 rows)."""
    return _read_all_rows()


def load_unsafe_user_prompts(*, include_needs_caution: bool = True) -> list[AegisRow]:
    """Return the unsafe user-message prompts, in file order.

    Filters to ``text_type == "user_message"`` and an unsafe ``labels_0``
    (see :func:`aegis_claim.categories.is_unsafe`). This mirrors the deepteam
    Aegis framework's attack set (``text_type == "user_message"`` and
    ``labels_0 != "Safe"``) when ``include_needs_caution`` is True.

    Args:
        include_needs_caution: When True (default), prompts whose only label is
            ``"Needs Caution"`` are included (deepteam-faithful, 236 rows on the
            pinned test split). When False, only the 13 harm categories count
            (180 rows).
    """
    return [
        row
        for row in _read_all_rows()
        if row["text_type"] == USER_MESSAGE_TEXT_TYPE
        and is_unsafe(row["labels_0"], include_needs_caution=include_needs_caution)
    ]


__all__ = [
    "AegisRow",
    "USER_MESSAGE_TEXT_TYPE",
    "load_all_rows",
    "load_unsafe_user_prompts",
]
