#!/usr/bin/env python3
"""Verify/refresh the vendored Snowball data and detector regexes vs garak.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "2233d51bedf6bfb634872ef9efbcd94eae38cc1a"
DATA_URL = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/data/{name}.json"
DET_URL = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/detectors/snowball.py"
DATA_FILES = ("graph_connectivity", "primality_testing", "senator_search")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "src/snowball_claim/data"
JUDGE = ROOT / "src/snowball_claim/judge.py"


def _get(url: str) -> str:
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def _regexes(text: str) -> set[str]:
    return set(re.findall(r'r"(\^\(\(.*?)"', text))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    drift = 0
    for name in DATA_FILES:
        up = _get(DATA_URL.format(commit=args.commit, name=name))
        dest = DATA_DIR / f"{name}.json"
        local = dest.read_text("utf-8") if dest.exists() else ""
        same = up == local
        if args.check:
            print(f"{'ok  ' if same else 'DIFF'} {name}.json")
            drift += 0 if same else 1
        elif same:
            print(f"unchanged {name}.json")
        else:
            dest.write_text(up, "utf-8")
            print(f"updated   {name}.json")

    up_rx = _regexes(_get(DET_URL.format(commit=args.commit)))
    local_rx = _regexes(JUDGE.read_text("utf-8"))
    rx_same = up_rx.issubset(local_rx) and up_rx
    print(f"{'ok  ' if rx_same else 'DIFF'} detector regexes")
    if args.check:
        drift += 0 if rx_same else 1
        if drift:
            print(f"\n{drift} item(s) differ from garak@{args.commit[:8]}", file=sys.stderr)
            return 1
    elif not rx_same:
        print("NOTE: detector regexes differ; edit judge.py by hand", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
