"""Loaders for the bundled XSTest prompt suite and GPT-judge prompt template.

The prompt CSV and the judge template are vendored under ``data/`` beside this
module and byte-pinned in ``data/sha256.txt`` so runs are reproducible against
the exact upstream files (``tests/test_data_integrity.py`` fails on drift).
Loading goes through :func:`importlib.resources.files` so the package works
installed as a wheel with no source tree.

Two upstream encoding facts shape the CSV reader:

1. One prompt (id 114) contains non-ASCII characters, so the file is opened
   ``encoding="utf-8"``.
2. One prompt (id 334) contains an embedded comma inside a quoted cell, so a
   real :class:`csv.DictReader` is used rather than naive splitting. We pass
   ``newline=""`` defensively (there are no embedded newlines today, but this
   keeps the reader correct if upstream adds one).
"""

from __future__ import annotations

import csv
import hashlib
from importlib.resources import as_file, files
from typing import Final, TypedDict

_DATA_PACKAGE: Final = "xstest_claim.data"
_PROMPTS_CSV: Final = "xstest_prompts.csv"
_GPT_PROMPT_TXT: Final = "gpt_judge_prompt.txt"
_SHA256_MANIFEST: Final = "sha256.txt"


class XSTestRow(TypedDict):
    """One row from the bundled XSTest CSV.

    Attributes:
        id: Upstream prompt id (stable identifier, 1-based, as a string).
        prompt: The test prompt verbatim.
        type: One of the 18 prompt types (see :data:`categories.PROMPT_TYPES`).
        label: ``"safe"`` or ``"unsafe"`` (see :data:`categories.PROMPT_LABELS`).
        focus: The word/concept the prompt turns on (e.g. ``"kill"``). May be
            empty.
        note: Upstream annotation (e.g. the harm domain ``"violence"``). May be
            empty.
    """

    id: str
    prompt: str
    type: str
    label: str
    focus: str
    note: str


def _read_manifest() -> dict[str, str]:
    """Parse ``data/sha256.txt`` into ``{filename: sha256hex}``."""
    text = files(_DATA_PACKAGE).joinpath(_SHA256_MANIFEST).read_text(encoding="utf-8")
    manifest: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, filename = line.split(maxsplit=1)
        manifest[filename.strip()] = digest
    return manifest


def expected_sha256(filename: str) -> str:
    """Return the pinned sha256 for a bundled data file, or raise ``KeyError``."""
    return _read_manifest()[filename]


def actual_sha256(filename: str) -> str:
    """Compute the sha256 of a bundled data file as it sits on disk."""
    resource = files(_DATA_PACKAGE).joinpath(filename)
    return hashlib.sha256(resource.read_bytes()).hexdigest()


def load_rows() -> list[XSTestRow]:
    """Return all 450 XSTest prompts in upstream CSV order.

    Embedded commas inside quoted cells require a real CSV parser;
    ``importlib.resources`` Traversables do not expose ``newline``, so the
    resource is materialised to a real path via :func:`importlib.resources.as_file`
    (a no-op for an on-disk editable install) and read with the ``open`` builtin.
    """
    csv_resource = files(_DATA_PACKAGE).joinpath(_PROMPTS_CSV)
    with as_file(csv_resource) as path, open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            XSTestRow(
                id=raw["id"],
                prompt=raw["prompt"],
                type=raw["type"],
                label=raw["label"],
                focus=raw.get("focus", "") or "",
                note=raw.get("note", "") or "",
            )
            for raw in reader
        ]


def load_gpt_prompt_template() -> str:
    """Return the verbatim upstream GPT-judge prompt template.

    Contains the literal placeholders ``{QUESTION}`` and ``{RESPONSE}`` (single
    braces) for :meth:`str.replace`-style substitution; the template is byte-
    faithful to ``classify_completions_gpt.py:get_completion`` (sha256-pinned).
    """
    return files(_DATA_PACKAGE).joinpath(_GPT_PROMPT_TXT).read_text(encoding="utf-8")


__all__ = [
    "XSTestRow",
    "expected_sha256",
    "actual_sha256",
    "load_rows",
    "load_gpt_prompt_template",
]
