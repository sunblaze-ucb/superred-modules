#!/usr/bin/env python3
"""Verify (or refresh) the vendored SafeClawBench Exec-Balanced files.

Re-fetches the pinned upstream commit from the SafeClawBench Hugging Face
dataset and diffs every vendored file against it, so the byte-for-byte vendoring
claim in ``ASSUMPTIONS.md`` stays honest.

    python scripts/sync_upstream.py --check    # exit 1 if anything drifted
    python scripts/sync_upstream.py --update    # copy upstream over the vendored tree

The vendored files are pure-stdlib; only ``git`` (with LFS smudge skipped) is
needed to fetch the pinned commit.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UPSTREAM_REPO = "https://huggingface.co/datasets/sairights/safeclawbench"
PINNED_COMMIT = "e6c29204c24a5910600aae854baae57a51586655"

_HERE = Path(__file__).resolve().parent.parent
_VENDOR = _HERE / "src" / "safeclawbench_exec_target" / "_vendor"

# vendored path (relative to _VENDOR)  ->  upstream path (relative to repo root)
FILES: dict[str, str] = {
    "executable/__init__.py": "executable/__init__.py",
    "executable/schema.py": "executable/schema.py",
    "executable/state.py": "executable/state.py",
    "executable/tools.py": "executable/tools.py",
    "executable/trajectory.py": "executable/trajectory.py",
    "executable/metrics.py": "executable/metrics.py",
    "executable/runner.py": "executable/runner.py",
    "executable/fixtures.py": "executable/fixtures.py",
    "executable/README.md": "executable/README.md",
    "executable/fixtures/exec_full_600.json": "executable/fixtures/exec_full_600.json",
    "executable/fixtures/tiny_subset.json": "executable/fixtures/tiny_subset.json",
    "defenses/defense_stack.py": "defenses/defense_stack.py",
}


def _clone(dest: Path) -> None:
    subprocess.run(
        ["git", "clone", "--quiet", UPSTREAM_REPO, str(dest)],
        check=True,
        env={"GIT_LFS_SKIP_SMUDGE": "1", "PATH": _env_path()},
    )
    subprocess.run(
        ["git", "-C", str(dest), "checkout", "--quiet", PINNED_COMMIT], check=True
    )


def _env_path() -> str:
    import os

    return os.environ.get("PATH", "/usr/bin:/bin")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="fail if vendored drifted")
    group.add_argument("--update", action="store_true", help="overwrite from upstream")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        upstream = Path(tmp) / "safeclawbench"
        _clone(upstream)

        drifted: list[str] = []
        for vend_rel, up_rel in FILES.items():
            vend = _VENDOR / vend_rel
            up = upstream / up_rel
            if not up.exists():
                print(f"MISSING upstream: {up_rel}")
                drifted.append(vend_rel)
                continue
            if args.update:
                vend.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(up, vend)
                continue
            if not vend.exists() or not filecmp.cmp(vend, up, shallow=False):
                print(f"DRIFT: {vend_rel}")
                drifted.append(vend_rel)

        if args.update:
            if drifted:
                print(
                    f"\n{len(drifted)} file(s) missing upstream at {PINNED_COMMIT[:9]}; "
                    "vendored tree left stale"
                )
                return 1
            print(f"updated {len(FILES)} vendored files from {PINNED_COMMIT[:9]}")
            return 0
        if drifted:
            print(f"\n{len(drifted)} vendored file(s) drifted from {PINNED_COMMIT[:9]}")
            return 1
        print(f"OK: all {len(FILES)} vendored files match {PINNED_COMMIT[:9]}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
