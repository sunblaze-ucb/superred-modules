#!/usr/bin/env python3
"""Verify (or refresh) what this package takes from upstream SafeClawBench.

Clones the pinned commit of the SafeClawBench Hugging Face dataset and checks
both kinds of upstream material, so the provenance claims in the README stay
honest:

- the data files vendored under ``src/safeclawbench_claim/data/`` must be
  byte-for-byte identical to upstream;
- the two prompts ported as Python constants, ``JUDGE_PROMPT`` (upstream
  ``evaluator/judge.py``) and ``AGENT_SYSTEM_PROMPT`` (upstream
  ``run_benchmark.py``), must equal upstream's strings exactly.

    python scripts/sync_upstream.py --check     # exit 1 if anything drifted
    python scripts/sync_upstream.py --update    # copy upstream data files over

Exit status: 0 when everything matches, 1 on drift, 2 when the pinned commit
could not be fetched. ``--update`` refreshes the data files only; a drifted
prompt constant has to be edited by hand. Only ``git`` (with LFS smudge
skipped) and the standard library are needed: both sides' prompts are read
with :mod:`ast`, so the package does not have to be importable.
"""

from __future__ import annotations

import argparse
import ast
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UPSTREAM_REPO = "https://huggingface.co/datasets/sairights/safeclawbench"
PINNED_COMMIT = "e6c29204c24a5910600aae854baae57a51586655"

_HERE = Path(__file__).resolve().parent.parent
_PKG = _HERE / "src" / "safeclawbench_claim"

# vendored path (relative to _PKG)  ->  upstream path (relative to repo root)
FILES: dict[str, str] = {
    "data/benchmark_v5_600.json": "benchmark_v5_600.json",
    "data/CITATION.cff": "CITATION.cff",
    "data/DATASET_LICENSE": "LICENSE",
}

# constant name  ->  (our module relative to _PKG, upstream module)
CONSTANTS: dict[str, tuple[str, str]] = {
    "JUDGE_PROMPT": ("judge.py", "evaluator/judge.py"),
    "AGENT_SYSTEM_PROMPT": ("loader.py", "run_benchmark.py"),
}


def _clone(dest: Path) -> None:
    # Smudge stays off for the checkout as well as the clone: upstream keeps
    # paper.pdf in LFS, and nothing here needs it.
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(
        ["git", "clone", "--quiet", UPSTREAM_REPO, str(dest)], check=True, env=env
    )
    subprocess.run(
        ["git", "-C", str(dest), "checkout", "--quiet", PINNED_COMMIT],
        check=True,
        env=env,
    )


def _string_constant(path: Path, name: str) -> str | None:
    """The string literal bound to ``name`` at module level in ``path``.

    Adjacent string literals are joined by the parser, so a parenthesised
    multi-line constant compares equal to upstream's triple-quoted one. Fails
    closed: ``None`` unless the module binds ``name`` exactly once and to a plain
    string literal, so a later rebinding (``X = X.replace(...)``) can never be
    mistaken for a match.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[ast.expr | None] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            values.append(node.value)
    if len(values) != 1:
        return None
    value = values[0]
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="fail if anything drifted")
    group.add_argument(
        "--update", action="store_true", help="overwrite data files from upstream"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        upstream = Path(tmp) / "safeclawbench"
        try:
            _clone(upstream)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"could not fetch {UPSTREAM_REPO} at {PINNED_COMMIT[:9]}: {exc}")
            return 2

        drifted: list[str] = []
        for vend_rel, up_rel in FILES.items():
            vend = _PKG / vend_rel
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

        for name, (ours_rel, up_rel) in CONSTANTS.items():
            up_path = upstream / up_rel
            theirs = _string_constant(up_path, name) if up_path.exists() else None
            if theirs is None:
                print(f"MISSING upstream: a single string literal {name} in {up_rel}")
                drifted.append(name)
            elif _string_constant(_PKG / ours_rel, name) != theirs:
                print(f"DRIFT: {name} ({ours_rel} vs upstream {up_rel})")
                drifted.append(name)

    if drifted:
        print(f"\n{len(drifted)} item(s) differ from {PINNED_COMMIT[:9]}")
        if args.update and any(name in CONSTANTS for name in drifted):
            print(
                "--update copies data files only; edit the drifted prompt constants by hand"
            )
        return 1
    if args.update:
        print(
            f"updated {len(FILES)} data files from {PINNED_COMMIT[:9]}; both prompts match"
        )
    else:
        print(
            f"OK: {len(FILES)} data files and {len(CONSTANTS)} prompts match {PINNED_COMMIT[:9]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
