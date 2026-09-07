#!/usr/bin/env python3
"""Verify/refresh the vendored PromptInject files against NVIDIA garak.

garak redistributes the PromptInject subset under garak/resources/promptinject/;
this module vendors the same files. --check compares byte-for-byte.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "2233d51bedf6bfb634872ef9efbcd94eae38cc1a"
RAW = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/resources/promptinject/{name}.py"
FILES = ("__init__", "_utils", "prompt_data", "prompting")
DEST = Path(__file__).resolve().parent.parent / "src/promptinject_claim/_vendor/promptinject"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    drift = 0
    for name in FILES:
        with urllib.request.urlopen(RAW.format(commit=args.commit, name=name)) as r:
            up = r.read().decode("utf-8")
        dest = DEST / f"{name}.py"
        local = dest.read_text("utf-8") if dest.exists() else ""
        same = up == local
        if args.check:
            print(f"{'ok  ' if same else 'DIFF'} {name}.py")
            drift += 0 if same else 1
        elif same:
            print(f"unchanged {name}.py")
        else:
            dest.write_text(up, "utf-8")
            print(f"updated   {name}.py")
    if args.check and drift:
        print(f"\n{drift} file(s) differ from garak@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
