#!/usr/bin/env python3
"""Synchronize a byte-exact L1B3RT4S snapshot from a local checkout.

This script deliberately accepts only the pinned commit.  Updating upstream is
a review event: change PINNED_COMMIT, inspect every corpus change, update
ASSUMPTIONS.md, then regenerate and rerun the parity tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PINNED_COMMIT = "64960b783249d36f76a48a33103cc4b168332b9b"
REPOSITORY = "https://github.com/elder-plinius/L1B3RT4S"

STORED_NAMES = {
    "!SHORTCUTS.json": "SHORTCUTS.json",
    "#MOTHERLOAD.txt": "MOTHERLOAD.txt",
    "*SPECIAL_TOKENS.json": "SPECIAL_TOKENS.json",
    "-MISCELLANEOUS-.mkd": "MISCELLANEOUS.mkd",
    "README.md": "UPSTREAM_README.md",
}

OMITTED = {
    "LICENSE": "copied to the module root as LICENSE instead of package data",
}


def _git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args],
        text=True,
    ).strip()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path, help="local L1B3RT4S checkout")
    args = parser.parse_args()
    checkout = args.checkout.resolve()

    commit = _git(checkout, "rev-parse", "HEAD")
    if commit != PINNED_COMMIT:
        raise SystemExit(
            f"checkout is at {commit}, expected pinned commit {PINNED_COMMIT}"
        )
    if _git(checkout, "status", "--porcelain"):
        raise SystemExit("upstream checkout must be clean")

    module_root = Path(__file__).resolve().parent.parent
    data_root = module_root / "src" / "libertas_optimizer" / "data"
    upstream_root = data_root / "upstream"
    upstream_root.mkdir(parents=True, exist_ok=True)

    tracked = _git(checkout, "ls-files").splitlines()
    root_files = sorted(path for path in tracked if "/" not in path)
    manifest_files: list[dict[str, object]] = []

    expected_stored: set[str] = set()
    for source_path in root_files:
        content = (checkout / source_path).read_bytes()
        omitted_reason = OMITTED.get(source_path)
        bundled = omitted_reason is None
        stored_path = STORED_NAMES.get(source_path, source_path) if bundled else None
        if stored_path is not None:
            expected_stored.add(stored_path)
            (upstream_root / stored_path).write_bytes(content)
        manifest_files.append(
            {
                "source_path": source_path,
                "stored_path": stored_path,
                "sha256": _sha256(content),
                "size": len(content),
                "bundled": bundled,
                "reason": omitted_reason,
            }
        )

    for existing in upstream_root.iterdir():
        if existing.is_file() and existing.name not in expected_stored:
            existing.unlink()

    manifest = {
        "repository": REPOSITORY,
        "commit": PINNED_COMMIT,
        "files": manifest_files,
    }
    (data_root / "upstream_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    shutil.copyfile(checkout / "LICENSE", module_root / "LICENSE")
    print(f"synced {sum(bool(item['bundled']) for item in manifest_files)} files")


if __name__ == "__main__":
    main()
