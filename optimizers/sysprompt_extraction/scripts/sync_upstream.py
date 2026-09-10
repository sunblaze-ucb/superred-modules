#!/usr/bin/env python3
"""Verify the vendored attack templates against garak.

``data/sysprompt_extraction/attacks.json`` is vendored byte-identical from
garak; this re-downloads it and diffs. The probe class itself is not ported
(it depends on `datasets`/HuggingFace to plant synthetic system prompts -- see
ASSUMPTIONS), so only the data file is verified. Nothing fetched over the
network is executed.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "2233d51bedf6bfb634872ef9efbcd94eae38cc1a"
RAW = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/data/sysprompt_extraction/attacks.json"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "src/sysprompt_extraction_optimizer/_vendor/garak_sysprompt/attacks.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    with urllib.request.urlopen(RAW.format(commit=args.commit)) as r:
        upstream = r.read().decode("utf-8")
    current = DEST.read_text(encoding="utf-8") if DEST.exists() else ""
    same = current == upstream

    if args.check:
        print(f"{'ok  ' if same else 'DIFF'} attacks.json")
        if not same:
            print(f"\nattacks.json differs from garak@{args.commit[:8]}", file=sys.stderr)
            return 1
        return 0
    if not same:
        DEST.write_text(upstream, encoding="utf-8")
        print("updated   attacks.json")
    else:
        print("unchanged attacks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
