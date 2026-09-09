#!/usr/bin/env python3
"""Verify the vendored ActorAttack template against Tencent AI-Infra-Guard.

Based on Tencent Zhuque Lab AI-Infra-Guard
(https://github.com/Tencent/AI-Infra-Guard).

The template is a byte-identical copy, so --check is a plain comparison. The
schema field names this module parses are additionally asserted against
upstream's schema.py.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "dd6bd54655c9ff5fb7351f4299b56916f09ec6da"
BASE = "https://raw.githubusercontent.com/Tencent/AI-Infra-Guard/{commit}/{path}"
ROOT = "AIG-PromptSecurity/deepteam/attacks/multi_turn/actor_attack"
DEST = (
    Path(__file__).resolve().parent.parent
    / "src/actor_attack_optimizer/_vendor/aig_actor_attack"
)
# Fields parsing.py reads out of upstream's pydantic models.
SCHEMA_FIELDS = (
    "actor_name", "relation_to_goal", "opening_question",
    "next_question", "is_final_probe", "classification", "rating",
)


def _get(path: str, commit: str) -> str:
    with urllib.request.urlopen(BASE.format(commit=commit, path=path)) as response:
        return response.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    drift = 0
    upstream = _get(f"{ROOT}/template.py", args.commit)
    dest = DEST / "template.py"
    current = dest.read_text(encoding="utf-8") if dest.exists() else ""
    same = current == upstream
    if args.check:
        print(f"{'ok  ' if same else 'DIFF'} template.py")
        drift += 0 if same else 1
    elif same:
        print("unchanged template.py")
    else:
        dest.write_text(upstream, encoding="utf-8")
        print("updated   template.py")

    schema = _get(f"{ROOT}/schema.py", args.commit)
    missing = [f for f in SCHEMA_FIELDS if f not in schema]
    ok = not missing
    print(f"{'ok  ' if ok else 'DIFF'} schema fields")
    if missing:
        print(f"     upstream schema no longer defines: {missing}", file=sys.stderr)
        drift += 1

    if args.check and drift:
        print(f"\n{drift} item(s) differ from AI-Infra-Guard@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
