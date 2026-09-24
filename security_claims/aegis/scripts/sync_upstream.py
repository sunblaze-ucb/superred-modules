#!/usr/bin/env python3
"""Verify (or refresh) the vendored Aegis test split against the pinned commit.

The authoritative benchmark is the NVIDIA *Aegis AI Content Safety Dataset 1.0*
(a.k.a. *Nemotron Content Safety Dataset V1*) on HuggingFace, CC-BY-4.0. We
vendor its **test split** two ways:

* ``_vendor/<PARQUET_FILENAME>`` — the upstream parquet, byte-identical to the
  file at the pinned revision. This is the provenance anchor: ``--check``
  byte-compares it against a fresh HuggingFace download.
* ``_vendor/aegis_test.csv`` — a deterministic CSV rendering of that parquet, so
  the claim can load the data with the stdlib :mod:`csv` module (no pandas, no
  pyarrow at runtime). :func:`parquet_bytes_to_csv_bytes` is the single source
  of truth for that rendering; ``--check`` regenerates the CSV from the vendored
  parquet and byte-compares it to the vendored CSV.

    python scripts/sync_upstream.py            # refresh both vendored files
    python scripts/sync_upstream.py --check     # verify byte-identical; exit 1 on drift
    python scripts/sync_upstream.py --manifest  # rewrite _vendor/SHA256SUMS

``_vendor/SHA256SUMS`` records the hashes so ``tests/test_vendor_integrity.py``
can verify the tree offline. Nothing fetched over the network is executed;
``pyarrow`` and ``huggingface_hub`` are imported lazily so the claim package
itself never depends on them.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pinned provenance (verified by the lead). The Aegis 1.0 repo is public
# (gated=False) and CC-BY-4.0. The revision is the dataset commit the vendored
# files were taken from; pinning guards against silent upstream drift.
# ---------------------------------------------------------------------------
REPO_ID = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"
REVISION = "bd96d862068e47630197de64eb91f8d1481ff3e0"
SPLIT = "test"
PARQUET_FILENAME = (
    "Content Moderation Extracted Annotations 02.08.24_test_release_0418_v1.parquet"
)
CSV_FILENAME = "aegis_test.csv"

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_ROOT = os.path.dirname(_HERE)


def _pkg_dir() -> str:
    src = os.path.join(_MODULE_ROOT, "src")
    pkgs = [d for d in sorted(os.listdir(src)) if os.path.isdir(os.path.join(src, d))]
    if len(pkgs) != 1:
        raise SystemExit(f"expected exactly one package under src/, found {pkgs!r}")
    return os.path.join(src, pkgs[0])


def _vendor_dir(pkg: str) -> str:
    return os.path.join(pkg, "_vendor")


def parquet_bytes_to_csv_bytes(parquet_bytes: bytes) -> bytes:
    """Render the Aegis parquet as a deterministic UTF-8 CSV.

    The rendering is fixed so it is reproducible on any machine:

    * columns are written in the parquet's own schema order, with a header row;
    * every value is rendered with ``str`` except ``None``/null, which becomes
      the empty string (the ``labels_4`` column is entirely null in this split);
    * ``\\n`` line terminator and ``QUOTE_MINIMAL`` quoting, so the stdlib
      :class:`csv.DictReader` in :mod:`aegis_claim.loader` round-trips the
      embedded newlines, commas and quotes in the ``text`` column.

    ``pyarrow`` is imported here (not at module top) so it is only needed when
    someone runs this dev script, never by the installed claim package.
    """
    import pyarrow.parquet as pq  # lazy: dev-only dependency

    table = pq.read_table(io.BytesIO(parquet_bytes))
    names = list(table.schema.names)
    columns = {name: table.column(name).to_pylist() for name in names}
    n = table.num_rows

    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(names)
    for i in range(n):
        writer.writerow(
            ["" if columns[name][i] is None else str(columns[name][i]) for name in names]
        )
    return buf.getvalue().encode("utf-8")


def _download_parquet() -> bytes:
    """Download the pinned test-split parquet from HuggingFace (public, CC-BY-4.0)."""
    from huggingface_hub import hf_hub_download  # lazy: dev-only dependency

    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=PARQUET_FILENAME,
        repo_type="dataset",
        revision=REVISION,
    )
    return Path(local).read_bytes()


def _vendored_files(pkg: str) -> list[str]:
    """Vendored data files whose hashes go in the manifest (relative to pkg)."""
    vendor = _vendor_dir(pkg)
    rels = []
    for name in (PARQUET_FILENAME, CSV_FILENAME):
        path = os.path.join(vendor, name)
        if os.path.isfile(path):
            rels.append(os.path.relpath(path, pkg))
    return sorted(rels)


def _write_manifest(pkg: str) -> int:
    lines = []
    for rel in _vendored_files(pkg):
        digest = hashlib.sha256(Path(pkg, rel).read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}\n")
    lines.sort(key=lambda s: s.split("  ", 1)[1])
    Path(_vendor_dir(pkg), "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    print(f"wrote manifest with {len(lines)} entries")
    return 0


def _refresh(pkg: str) -> int:
    vendor = _vendor_dir(pkg)
    os.makedirs(vendor, exist_ok=True)
    parquet_bytes = _download_parquet()
    Path(vendor, PARQUET_FILENAME).write_bytes(parquet_bytes)
    Path(vendor, CSV_FILENAME).write_bytes(parquet_bytes_to_csv_bytes(parquet_bytes))
    print(
        f"refreshed {PARQUET_FILENAME!r} and {CSV_FILENAME!r} "
        f"from {REPO_ID}@{REVISION[:8]} (split={SPLIT})"
    )
    return 0


def _check(pkg: str) -> int:
    vendor = _vendor_dir(pkg)
    drift = 0

    vendored_parquet = Path(vendor, PARQUET_FILENAME)
    if not vendored_parquet.is_file():
        print(f"MISSING vendored parquet {PARQUET_FILENAME!r}")
        return 1

    upstream = _download_parquet()
    if vendored_parquet.read_bytes() != upstream:
        print(f"DIFF {PARQUET_FILENAME!r} (vendored parquet != {REPO_ID}@{REVISION[:8]})")
        drift += 1

    # The CSV is a deterministic rendering of the vendored parquet.
    regenerated = parquet_bytes_to_csv_bytes(vendored_parquet.read_bytes())
    vendored_csv = Path(vendor, CSV_FILENAME)
    if not vendored_csv.is_file():
        print(f"MISSING vendored csv {CSV_FILENAME!r}")
        drift += 1
    elif vendored_csv.read_bytes() != regenerated:
        print(f"DIFF {CSV_FILENAME!r} (not the deterministic rendering of the parquet)")
        drift += 1

    if drift:
        print(f"\n{drift} file(s) drifted from {REPO_ID}@{REVISION[:8]}")
        return 1
    print(f"ok: vendored test split byte-identical to {REPO_ID}@{REVISION[:8]}")
    return 0


def main() -> int:
    pkg = _pkg_dir()
    if "--manifest" in sys.argv:
        return _write_manifest(pkg)
    if "--check" in sys.argv:
        return _check(pkg)
    return _refresh(pkg)


if __name__ == "__main__":
    raise SystemExit(main())
