#!/usr/bin/env python3
"""Verify the vendored tense templates against Tencent AI-Infra-Guard.

Based on Tencent Zhuque Lab AI-Infra-Guard
(https://github.com/Tencent/AI-Infra-Guard).

template.py is a byte-identical copy, so --check is a plain comparison.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "dd6bd54655c9ff5fb7351f4299b56916f09ec6da"
URL = (
    "https://raw.githubusercontent.com/Tencent/AI-Infra-Guard/{commit}/"
    "AIG-PromptSecurity/deepteam/attacks/single_turn/jailbroken/template.py"
)
DEST = (
    Path(__file__).resolve().parent.parent
    / "src/jailbroken_optimizer/_vendor/aig_jailbroken/template.py"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    with urllib.request.urlopen(URL.format(commit=args.commit)) as response:
        upstream = response.read().decode("utf-8")
    current = DEST.read_text(encoding="utf-8") if DEST.exists() else ""
    same = current == upstream
    if args.check:
        print(f"{'ok  ' if same else 'DIFF'} template.py")
        if not same:
            print(f"\ntemplate.py differs from AI-Infra-Guard@{args.commit[:8]}", file=sys.stderr)
            return 1
    elif same:
        print("unchanged template.py")
    else:
        DEST.write_text(upstream, encoding="utf-8")
        print("updated   template.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
