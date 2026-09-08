#!/usr/bin/env python3
"""Verify/refresh the vendored Bad Likert Judge templates against DeepTeam.

Both files are byte-identical copies, so `--check` is a plain comparison. The
vendored template keeps upstream's absolute import of `BaseMultiTurnTemplate`;
`_vendor/loader.py` satisfies it without editing the file, which is what lets
this check stay a byte comparison.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "dc148aad62f71330cfec7121d6afb4c620dfa683"
BASE = "https://raw.githubusercontent.com/confident-ai/deepteam/{commit}/{path}"
FILES = {
    "template.py": "deepteam/attacks/multi_turn/bad_likert_judge/template.py",
    "base_template.py": "deepteam/attacks/multi_turn/base_template.py",
}
DEST = (
    Path(__file__).resolve().parent.parent
    / "src/bad_likert_judge_optimizer/_vendor/deepteam_blj"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    drift = 0
    for local, remote in FILES.items():
        url = BASE.format(commit=args.commit, path=remote)
        with urllib.request.urlopen(url) as response:
            upstream = response.read().decode("utf-8")
        dest = DEST / local
        current = dest.read_text(encoding="utf-8") if dest.exists() else ""
        same = current == upstream
        if args.check:
            print(f"{'ok  ' if same else 'DIFF'} {local}")
            drift += 0 if same else 1
        elif same:
            print(f"unchanged {local}")
        else:
            dest.write_text(upstream, encoding="utf-8")
            print(f"updated   {local}")

    if args.check and drift:
        print(
            f"\n{drift} file(s) differ from deepteam@{args.commit[:8]}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
