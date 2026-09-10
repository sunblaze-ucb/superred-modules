#!/usr/bin/env python3
"""Verify the vendored ANSI resources and prompt constants against garak.

Two things are pinned:

* ``_vendor/garak_ansi/ansi.py`` is a byte-identical copy of garak's
  ``resources/ansi.py``;
* ``data.json`` holds the four prompt-stub lists that live in
  ``probes/ansiescape.py``, which cannot be imported (it pulls in garak's
  probe machinery), so they are re-extracted here by AST.

Nothing fetched over the network is executed.

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
ANSI = ROOT / "src/ansi_escape_optimizer/_vendor/garak_ansi/ansi.py"
DATA = ROOT / "src/ansi_escape_optimizer/data.json"
KEYS = ("ASKS", "HIGH_LEVEL_TASKS", "REPEAT_STUBS", "UNESCAPE_STUBS")


def _get(path: str, commit: str) -> str:
    with urllib.request.urlopen(RAW.format(commit=commit, path=path)) as r:
        return r.read().decode("utf-8")


def _extract(src: str) -> dict:
    return {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(src).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in KEYS
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    drift = 0

    up_ansi = _get("resources/ansi.py", args.commit)
    cur = ANSI.read_text(encoding="utf-8") if ANSI.exists() else ""
    same = cur == up_ansi
    if args.check:
        print(f"{'ok  ' if same else 'DIFF'} resources/ansi.py")
        drift += 0 if same else 1
    elif not same:
        ANSI.write_text(up_ansi, encoding="utf-8")
        print("updated   resources/ansi.py")
    else:
        print("unchanged resources/ansi.py")

    upstream = _extract(_get("probes/ansiescape.py", args.commit))
    missing = [k for k in KEYS if k not in upstream]
    if missing:
        print(
            f"extraction failed: {missing} not found in garak@{args.commit[:8]}; "
            "upstream has probably moved. data.json is left untouched.",
            file=sys.stderr,
        )
        return 1

    local = json.loads(DATA.read_text("utf-8")) if DATA.exists() else {}
    for k in KEYS:
        same = upstream.get(k) == local.get(k)
        print(f"{'ok  ' if same else 'DIFF'} {k}")
        drift += 0 if same else 1

    if not args.check and drift:
        DATA.write_text(
            json.dumps(upstream, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("updated data.json")
        return 0
    if args.check and drift:
        print(f"\n{drift} item(s) differ from garak@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
