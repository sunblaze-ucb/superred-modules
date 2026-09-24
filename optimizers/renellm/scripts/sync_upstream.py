#!/usr/bin/env python3
"""Verify (or refresh) vendored ReNeLLM files against the pinned commit.

Vendored files live under ``src/<pkg>/_vendor/renellm/<path>`` mirroring the
upstream repo layout (e.g. ``_vendor/renellm/utils/data_utils.py`` maps to
upstream ``utils/data_utils.py``). This script maps each back to its upstream
path and byte-compares.

    python scripts/sync_upstream.py            # refresh vendored files
    python scripts/sync_upstream.py --check     # verify byte-identical; exit 1 on drift
    python scripts/sync_upstream.py --manifest  # rewrite _vendor/SHA256SUMS

The pinned commit is the authoritative source; ``_vendor/SHA256SUMS`` records
the hashes so ``tests/test_vendor_integrity.py`` can verify the tree offline.
Nothing fetched over the network is executed. Only ReNeLLM's torch-free
rewrite/nest/judge helpers are vendored; ``utils/llm_completion_utils.py`` is
intentionally not vendored (it is replaced by the in-module LLM shim).
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

PINNED_COMMIT = "a61c39ea4e5311a5fbfaee9869e5325783d1522f"
REPO = "NJUNLP/ReNeLLM"
VENDOR_SUBDIR = "renellm"

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_ROOT = os.path.dirname(_HERE)


def _pkg_dir() -> str:
    src = os.path.join(_MODULE_ROOT, "src")
    pkgs = [d for d in sorted(os.listdir(src)) if os.path.isdir(os.path.join(src, d))]
    if len(pkgs) != 1:
        raise SystemExit(f"expected exactly one package under src/, found {pkgs!r}")
    return os.path.join(src, pkgs[0])


def _pairs(pkg: str) -> list[tuple[str, str]]:
    """Return (absolute local path, upstream repo path) for every vendored file."""
    pairs: list[tuple[str, str]] = []
    vendor_root = os.path.join(pkg, "_vendor", VENDOR_SUBDIR)
    for root, _dirs, names in os.walk(vendor_root):
        if "__pycache__" in root:
            continue
        for name in names:
            local = os.path.join(root, name)
            upstream = os.path.relpath(local, vendor_root)  # e.g. utils/data_utils.py
            pairs.append((local, upstream))
    return sorted(pairs)


def _download() -> tarfile.TarFile:
    url = f"https://github.com/{REPO}/archive/{PINNED_COMMIT}.tar.gz"
    with urllib.request.urlopen(url, timeout=120) as resp:  # pinned https tarball
        data = resp.read()
    return tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")


def _write_manifest(pkg: str, pairs: list[tuple[str, str]]) -> int:
    lines = []
    for local, _up in pairs:
        rel = os.path.relpath(local, pkg)
        digest = hashlib.sha256(Path(local).read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}\n")
    lines.sort(key=lambda s: s.split("  ", 1)[1])
    Path(pkg, "_vendor", "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    print(f"wrote manifest with {len(lines)} entries")
    return 0


def main() -> int:
    pkg = _pkg_dir()
    pairs = _pairs(pkg)
    if "--manifest" in sys.argv:
        return _write_manifest(pkg, pairs)

    check = "--check" in sys.argv
    tf = _download()
    root = tf.getnames()[0].split("/")[0]
    names = set(tf.getnames())
    drift = written = 0
    for local, upstream in pairs:
        full = f"{root}/{upstream}"
        if full not in names:
            print(f"MISSING upstream {upstream}")
            drift += 1
            continue
        member = tf.extractfile(full)
        up_bytes = member.read() if member else b""
        cur = Path(local).read_bytes() if os.path.exists(local) else None
        if cur == up_bytes:
            continue
        if check:
            print(f"DIFF {upstream}")
            drift += 1
        else:
            os.makedirs(os.path.dirname(local), exist_ok=True)
            Path(local).write_bytes(up_bytes)
            written += 1
    if check:
        if drift:
            print(f"\n{drift} file(s) drifted from {REPO}@{PINNED_COMMIT[:8]}")
            return 1
        print(f"ok: {len(pairs)} vendored file(s) byte-identical to {REPO}@{PINNED_COMMIT[:8]}")
        return 0
    print(f"refreshed {written} file(s) from {REPO}@{PINNED_COMMIT[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
