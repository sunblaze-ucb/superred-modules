#!/usr/bin/env python3
"""Verify (or refresh) vendored MT-JailBench files against the pinned commit.

Vendored payloads live under ``src/<pkg>/_vendor/mtjb/<path>`` (mirroring the
upstream repo layout) and, for the claim, ``src/<pkg>/data/<file>``. This
script maps each back to its upstream path and byte-compares.

    python scripts/sync_upstream.py            # refresh vendored files
    python scripts/sync_upstream.py --check     # verify byte-identical; exit 1 on drift
    python scripts/sync_upstream.py --manifest  # rewrite _vendor/SHA256SUMS

The pinned commit is the authoritative source; ``_vendor/SHA256SUMS`` records
the hashes so ``tests/test_vendor_integrity.py`` can verify the tree offline.
Nothing fetched over the network is executed.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

PINNED_COMMIT = "cb8184e0a9ad7a65c86a00e2a38c57a5f58043f0"
REPO = "SafetyArena/mt-jailbench"

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
    vendor_mtjb = os.path.join(pkg, "_vendor", "mtjb")
    for root, _dirs, names in os.walk(vendor_mtjb):
        for name in names:
            if name in {"SHA256SUMS", "__init__.py"} and root == vendor_mtjb:
                continue  # authored namespace files, not vendored
            if "__pycache__" in root:
                continue
            local = os.path.join(root, name)
            upstream = os.path.relpath(local, vendor_mtjb)  # e.g. engine/attacks/coa/...
            pairs.append((local, upstream))
    data_dir = os.path.join(pkg, "data")
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if name.endswith((".txt",)):  # sha256 manifests, not upstream payloads
                continue
            local = os.path.join(data_dir, name)
            if os.path.isfile(local):
                pairs.append((local, f"data/{name}"))
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
