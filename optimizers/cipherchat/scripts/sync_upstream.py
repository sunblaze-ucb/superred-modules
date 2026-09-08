#!/usr/bin/env python3
"""Verify the vendored CipherChat files against RobustNLP/CipherChat.

Both files (the ciphers and the prompt corpus) are byte-identical copies, so
--check is a plain comparison.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "6fce7891a9a072b933f13bba7f58651577499fb5"
BASE = "https://raw.githubusercontent.com/RobustNLP/CipherChat/{commit}/{name}"
FILES = ("encode_experts.py", "prompts_and_demonstrations.py")
DEST = (
    Path(__file__).resolve().parent.parent
    / "src/cipherchat_optimizer/_vendor/cipherchat"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    drift = 0
    for name in FILES:
        with urllib.request.urlopen(BASE.format(commit=args.commit, name=name)) as r:
            upstream = r.read().decode("utf-8")
        dest = DEST / name
        current = dest.read_text(encoding="utf-8") if dest.exists() else ""
        same = current == upstream
        if args.check:
            print(f"{'ok  ' if same else 'DIFF'} {name}")
            drift += 0 if same else 1
        elif same:
            print(f"unchanged {name}")
        else:
            dest.write_text(upstream, encoding="utf-8")
            print(f"updated   {name}")

    if args.check and drift:
        print(f"\n{drift} file(s) differ from CipherChat@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
