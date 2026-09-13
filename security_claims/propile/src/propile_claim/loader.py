"""Loaders for the bundled ProPILE prompt templates and PII records.

``prompt_templates.tsv`` is vendored byte-faithfully from garak (Apache-2.0) and
sha256-pinned. ``sample_pii.jsonl`` is a small SYNTHETIC dataset authored for this
module (public-domain names, ``example.*`` domains, reserved ``555-01xx`` numbers)
so the claim is runnable and testable with **no real PII**. Point ``pii_data_path``
at your own JSONL to evaluate against data you have a lawful basis to test.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import as_file, files
from pathlib import Path
from typing import Final

_DATA_PACKAGE: Final = "propile_claim.data"
_TEMPLATES_TSV: Final = "prompt_templates.tsv"
_SAMPLE_PII: Final = "sample_pii.jsonl"
_SHA_MANIFEST: Final = "sha256.txt"


def _read_manifest() -> dict[str, str]:
    text = files(_DATA_PACKAGE).joinpath(_SHA_MANIFEST).read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line:
            digest, filename = line.split(maxsplit=1)
            out[filename.strip()] = digest
    return out


def expected_sha256(filename: str) -> str:
    return _read_manifest()[filename]


def actual_sha256(filename: str) -> str:
    return hashlib.sha256(files(_DATA_PACKAGE).joinpath(filename).read_bytes()).hexdigest()


def load_templates() -> dict[str, list[str]]:
    """Return ``{category: [template, ...]}`` from the bundled TSV.

    Ported from ``PIILeakProbeMixin._load_templates`` — a tab-delimited file with
    ``category`` and ``template`` columns.
    """
    import csv

    resource = files(_DATA_PACKAGE).joinpath(_TEMPLATES_TSV)
    templates: dict[str, list[str]] = {}
    with as_file(resource) as path, open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            templates.setdefault(row["category"], []).append(row["template"])
    return templates


def load_pii_records(pii_data_path: str | Path | None = None) -> list[dict[str, str]]:
    """Return PII records from a JSONL file.

    Ported from ``PIILeakProbeMixin._load_pii_data``: one JSON object per line.
    Defaults to the bundled synthetic ``sample_pii.jsonl``; pass ``pii_data_path``
    to use your own dataset (records with ``name`` plus any of ``email``, ``phone``,
    ``address``, ``employer``, ``university``, ``organization``, relationship fields).
    """
    records: list[dict[str, str]] = []
    if pii_data_path is not None:
        with open(pii_data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    resource = files(_DATA_PACKAGE).joinpath(_SAMPLE_PII)
    for line in resource.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


__all__ = [
    "load_templates",
    "load_pii_records",
    "expected_sha256",
    "actual_sha256",
]
