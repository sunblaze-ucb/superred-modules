"""HarmBench behavior CSV loader.

Reads bundled CSVs from package data via :mod:`importlib.resources` so it
works whether the package is installed editable or as a zip-built wheel.

Each row is returned as a plain dict with the upstream column set plus a
synthetic ``_split`` key indicating which split the row was loaded from.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from importlib import resources
from typing import Literal

# Verbatim upstream column names. We do not normalize keys.
EXPECTED_COLUMNS: frozenset[str] = frozenset(
    {"Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID"}
)

VALID_SPLITS: tuple[str, ...] = ("test", "val")
VALID_FUNCTIONAL: frozenset[str] = frozenset({"standard", "contextual", "copyright"})

# (semantic categories aren't enforced; we accept anything upstream emits.)


@contextmanager
def _bundled_csv_handle(split: str) -> Iterator[csv.DictReader[str]]:
    """Yield a ``csv.DictReader`` over the bundled CSV for ``split``.

    Uses :func:`importlib.resources.as_file` so the resource is
    materialized to a real on-disk path inside the context, which lets
    us ``open()`` it. This works for both editable installs (where the
    resource is already a filesystem file) and zip-built wheels (where
    ``as_file`` extracts the resource to a temporary location for the
    duration of the ``with`` block).

    Stringifying the ``Traversable`` from ``resources.files()`` directly
    is unsafe: for zip-bundled wheels the resulting path is not openable
    by the standard ``open()`` builtin.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS!r}, got {split!r}")
    filename = f"harmbench_behaviors_text_{split}.csv"
    resource = resources.files("harmbench_claim") / "data" / filename
    with resources.as_file(resource) as real_path:
        with open(real_path, encoding="utf-8") as fh:
            yield csv.DictReader(fh)


def load_behaviors(
    *,
    split: Literal["test", "val"],
    functional_categories: Iterable[str] | None = None,
    semantic_categories: Iterable[str] | None = None,
    csv_path: str | None = None,
) -> list[dict[str, str]]:
    """Load HarmBench behavior rows, optionally filtered.

    Args:
        split: Which bundled split to read. Ignored if ``csv_path`` is
            provided, in which case ``_split`` on each row is set to
            ``"custom"``.
        functional_categories: Restrict to these functional categories.
            Each must be in ``{"standard", "contextual", "copyright"}``.
            ``None`` keeps all.
        semantic_categories: Restrict to these semantic categories
            (any string upstream emits). ``None`` keeps all.
        csv_path: Optional override path to a CSV with the upstream
            schema. Useful for using your own behavior set without
            rebuilding the wheel.

    Returns:
        List of row dicts in CSV order (deterministic). Each dict carries
        the original column keys plus a synthetic ``_split`` key.

    Raises:
        ValueError: If the split is invalid, the CSV is missing required
            columns, or filters select zero rows.
        FileNotFoundError: If ``csv_path`` doesn't exist.
    """
    if functional_categories is not None:
        wanted_fc = set(functional_categories)
        unknown = wanted_fc - VALID_FUNCTIONAL
        if unknown:
            raise ValueError(
                f"unknown functional_categories: {sorted(unknown)!r} "
                f"(valid: {sorted(VALID_FUNCTIONAL)!r})"
            )
    else:
        wanted_fc = None

    wanted_sc = set(semantic_categories) if semantic_categories is not None else None

    if csv_path is not None:
        source_label = repr(csv_path)
        with open(csv_path, encoding="utf-8") as fh:
            rows = _read_rows(csv.DictReader(fh), source_repr=source_label)
        split_label = "custom"
    else:
        source_label = f"bundled split={split!r}"
        with _bundled_csv_handle(split) as reader:
            rows = _read_rows(reader, source_repr=source_label)
        split_label = split

    out: list[dict[str, str]] = []
    for row in rows:
        if wanted_fc is not None and row["FunctionalCategory"] not in wanted_fc:
            continue
        if wanted_sc is not None and row["SemanticCategory"] not in wanted_sc:
            continue
        row["_split"] = split_label
        out.append(row)

    if not out:
        raise ValueError(
            "filters selected zero rows: "
            f"source={source_label}, "
            f"functional_categories={list(functional_categories) if functional_categories else None!r}, "
            f"semantic_categories={list(semantic_categories) if semantic_categories else None!r}"
        )

    return out


def _read_rows(
    reader: csv.DictReader[str], *, source_repr: str,
) -> list[dict[str, str]]:
    """Validate header and materialize all rows from a DictReader.

    Pulled out as a helper so both branches in :func:`load_behaviors`
    (bundled vs ``csv_path``) share the same validation.
    """
    if reader.fieldnames is None:
        raise ValueError(f"empty or unreadable CSV: {source_repr}")
    missing = EXPECTED_COLUMNS - set(reader.fieldnames)
    if missing:
        raise ValueError(
            f"CSV {source_repr} is missing required columns: {sorted(missing)!r}"
        )
    return [dict(r) for r in reader]
