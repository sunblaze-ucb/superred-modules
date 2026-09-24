"""Offline vendor byte-identity check.

Every file listed in ``_vendor/SHA256SUMS`` must still hash to its recorded
value. The manifest was generated from the Microsoft PyRIT tree at the pinned
commit ``2016c4a8`` (see ``scripts/sync_upstream.py``), so this guards the
vendored Skeleton Key seed prompts against accidental edits without needing
network access. The network byte-compare against the pinned commit lives in
``scripts/sync_upstream.py --check`` (run in CI, not the offline battery).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"


def _pkg_dir() -> Path:
    pkgs = [p for p in _SRC.iterdir() if p.is_dir() and (p / "_vendor").is_dir()]
    assert len(pkgs) == 1, f"expected one package with _vendor/, found {pkgs!r}"
    return pkgs[0]


def _manifest(pkg: Path) -> list[tuple[str, str]]:
    lines = (pkg / "_vendor" / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        out.append((digest, rel))
    return out


def test_manifest_is_not_empty() -> None:
    entries = _manifest(_pkg_dir())
    assert entries, "SHA256SUMS is empty; no vendored files are pinned"


def test_vendored_files_match_pinned_hashes() -> None:
    pkg = _pkg_dir()
    mismatches: list[str] = []
    for digest, rel in _manifest(pkg):
        path = pkg / rel
        if not path.is_file():
            mismatches.append(f"MISSING {rel}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            mismatches.append(f"DRIFT {rel} ({actual[:12]} != {digest[:12]})")
    assert not mismatches, "vendored files drifted from the pinned commit:\n  " + "\n  ".join(
        mismatches
    )
