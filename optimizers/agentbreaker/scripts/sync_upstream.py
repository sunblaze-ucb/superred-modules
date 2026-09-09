#!/usr/bin/env python3
"""Verify/refresh the vendored AgentBreaker prompts against NVIDIA garak.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "2233d51bedf6bfb634872ef9efbcd94eae38cc1a"
RAW = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/data/agent_breaker/prompts.yaml"
DEST = (
    Path(__file__).resolve().parent.parent
    / "src/agentbreaker_optimizer/data/upstream/prompts.yaml"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    with urllib.request.urlopen(RAW.format(commit=args.commit)) as r:
        upstream = r.read().decode("utf-8")
    local = DEST.read_text(encoding="utf-8") if DEST.exists() else ""
    same = local == upstream
    if args.check:
        print(f"{'ok  ' if same else 'DIFF'} prompts.yaml")
        if not same:
            print(f"differs from garak@{args.commit[:8]}", file=sys.stderr)
            return 1
        return 0
    DEST.write_text(upstream, encoding="utf-8")
    print("unchanged prompts.yaml" if same else "updated   prompts.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
