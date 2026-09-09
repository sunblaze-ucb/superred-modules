#!/usr/bin/env python3
"""Verify the vendored CodeAttack templates + tokenisation against upstream.

Checks two things: the 3 Python template files are byte-identical to upstream,
and render() reproduces upstream's own shipped data (data_python_{list,string}
_full.json) byte-for-byte. (The stack shipped data is stale upstream -- see
ASSUMPTIONS.md -- so stack is checked only for template identity.)

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "6777ed03b11567a91908f23bb8ccedca6103772c"
BASE = "https://raw.githubusercontent.com/renqibing/CodeAttack/{commit}"
TPL = "/src/codeattack/prompt_templates/{name}.txt"
DATA = "/prompts/data_python_{v}_full.json"
TEMPLATES = (
    "code_python_list",
    "code_python_stack",
    "code_python_string",
    "code_python_list_plus",
    "code_python_stack_plus",
    "code_python_string_plus",
    "code_C_string",
    "code_go_string",
)
ROOT = Path(__file__).resolve().parent.parent
TPL_DIR = ROOT / "src/codeattack_optimizer/data/upstream"


def _get(url: str) -> str:
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    drift = 0
    for name in TEMPLATES:
        up = _get(BASE.format(commit=args.commit) + TPL.format(name=name))
        dest = TPL_DIR / f"{name}.txt"
        local = dest.read_text("utf-8") if dest.exists() else ""
        same = up == local
        if args.check:
            print(f"{'ok  ' if same else 'DIFF'} {name}.txt")
            drift += 0 if same else 1
        elif not same:
            dest.write_text(up, "utf-8")
            print(f"updated   {name}.txt")
        else:
            print(f"unchanged {name}.txt")

    # render() vs upstream shipped data (list, string are consistent upstream)
    sys.path.insert(0, str(ROOT / "src"))
    from codeattack_optimizer.codeattack import render

    for v in ("list", "string"):
        d = json.loads(_get(BASE.format(commit=args.commit) + DATA.format(v=v)))
        ok = sum(
            1 for r in d
            if render(r["plain_attack"], f"python_{v}") == r["code_wrapped_plain_attack"]
        )
        good = ok == len(d)
        print(f"{'ok  ' if good else 'DIFF'} render python_{v} vs shipped data ({ok}/{len(d)})")
        if args.check:
            drift += 0 if good else 1

    if args.check and drift:
        print(f"\n{drift} item(s) differ from CodeAttack@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
