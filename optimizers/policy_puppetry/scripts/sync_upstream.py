#!/usr/bin/env python3
"""Verify the vendored Dr. House template against NVIDIA garak.

The template lives inline in ``garak/probes/doctor.py`` (as
``Bypass.templates[0]``), so this extracts it from upstream source at a pinned
commit and compares byte-for-byte with the vendored copy.

    python scripts/sync_upstream.py [--commit SHA] [--check]

``--check`` verifies without writing.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "2233d51bedf6bfb634872ef9efbcd94eae38cc1a"
RAW = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/probes/doctor.py"

DEST = (
    Path(__file__).resolve().parent.parent
    / "src/policy_puppetry_optimizer/data/upstream/bypass_template.txt"
)


def extract(commit: str) -> str:
    with urllib.request.urlopen(RAW.format(commit=commit)) as response:
        source = response.read().decode("utf-8")
    match = re.search(r'self\.templates = \[\s*"""(.*?)""",\s*\]', source, re.S)
    if match is None:
        raise SystemExit("could not locate Bypass.templates[0] in upstream doctor.py")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument("--check", action="store_true", help="verify only")
    args = parser.parse_args()

    upstream = extract(args.commit)
    local = DEST.read_text(encoding="utf-8") if DEST.exists() else ""
    same = local == upstream

    if args.check:
        print(f"{'ok  ' if same else 'DIFF'} bypass_template.txt")
        if not same:
            print(
                f"template differs from garak@{args.commit[:8]}", file=sys.stderr
            )
            return 1
        return 0

    if same:
        print("unchanged bypass_template.txt")
    else:
        DEST.write_text(upstream, encoding="utf-8")
        print("updated   bypass_template.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
