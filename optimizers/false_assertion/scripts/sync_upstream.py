#!/usr/bin/env python3
"""Verify the vendored false-claims data and prefixes against garak.

``data/truefalse_falseclaims_50.txt`` is vendored byte-identical and diffed.
The three ``prefixes`` live inside ``FalseAssertion.__init__`` in
``probes/misleading.py`` (which cannot be imported), so they are re-extracted
by AST and compared with ``data.json``. Nothing fetched over the network is
executed.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "2233d51bedf6bfb634872ef9efbcd94eae38cc1a"
RAW = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/{path}"
ROOT = Path(__file__).resolve().parent.parent
CLAIMS = ROOT / "src/false_assertion_optimizer/_vendor/garak_data/truefalse_falseclaims_50.txt"
DATA = ROOT / "src/false_assertion_optimizer/data.json"


def _get(path: str, commit: str) -> str:
    with urllib.request.urlopen(RAW.format(commit=commit, path=path)) as r:
        return r.read().decode("utf-8")


def _extract_prefixes(src: str) -> list | None:
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "prefixes":
            return list(ast.literal_eval(n.value))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    drift = 0

    up_claims = _get("data/truefalse_falseclaims_50.txt", args.commit)
    cur = CLAIMS.read_text(encoding="utf-8") if CLAIMS.exists() else ""
    same = cur == up_claims
    print(f"{'ok  ' if same else 'DIFF'} truefalse_falseclaims_50.txt")
    if not same:
        drift += 1
        if not args.check:
            CLAIMS.write_text(up_claims, encoding="utf-8")
            print("updated   truefalse_falseclaims_50.txt")

    up_prefixes = _extract_prefixes(_get("probes/misleading.py", args.commit))
    if up_prefixes is None:
        print(
            f"extraction failed: prefixes not found in garak@{args.commit[:8]}; "
            "upstream has probably moved. data.json is left untouched.",
            file=sys.stderr,
        )
        return 1
    local = json.loads(DATA.read_text("utf-8")) if DATA.exists() else {}
    same = up_prefixes == local.get("prefixes")
    print(f"{'ok  ' if same else 'DIFF'} prefixes")
    if not same:
        drift += 1
        if not args.check:
            DATA.write_text(
                json.dumps({"prefixes": up_prefixes}, indent=1) + "\n", encoding="utf-8"
            )
            print("updated data.json")

    if args.check and drift:
        print(f"\n{drift} item(s) differ from garak@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
