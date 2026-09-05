#!/usr/bin/env python3
"""Verify/refresh the vendored CyberSecEval PI data and judge prompt vs PurpleLlama.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "4be64c3a24442b51c76175e6ec67722cc3f5fe38"
BASE = "https://raw.githubusercontent.com/meta-llama/PurpleLlama/{commit}"
DATA = "/CybersecurityBenchmarks/datasets/prompt_injection/{name}"
JUDGE = "/CybersecurityBenchmarks/benchmark/prompt_injection_benchmark.py"
DATA_FILES = (
    "prompt_injection.json",
    "prompt_injection_multilingual_machine_translated.json",
)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "src/cyberseceval_pi_claim/data"
JUDGE_SRC = ROOT / "src/cyberseceval_pi_claim/judge.py"


def _get(url: str) -> str:
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def _judge_prompt(text: str) -> str:
    m = re.search(r'PROMPT_JUDGE_RESPONSE = (""".*?""")', text, re.S)
    if not m:
        raise SystemExit("could not find PROMPT_JUDGE_RESPONSE")
    return m.group(1)[3:-3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    drift = 0
    for name in DATA_FILES:
        up = _get(BASE.format(commit=args.commit) + DATA.format(name=name))
        dest = DATA_DIR / name
        local = dest.read_text("utf-8") if dest.exists() else ""
        same = up == local
        if args.check:
            print(f"{'ok  ' if same else 'DIFF'} {name}")
            drift += 0 if same else 1
        elif same:
            print(f"unchanged {name}")
        else:
            dest.write_text(up, "utf-8")
            print(f"updated   {name}")
    up_prompt = _judge_prompt(_get(BASE.format(commit=args.commit) + JUDGE))
    local_prompt = _judge_prompt(JUDGE_SRC.read_text("utf-8"))
    same = up_prompt == local_prompt
    print(f"{'ok  ' if same else 'DIFF'} PROMPT_JUDGE_RESPONSE")
    if args.check:
        drift += 0 if same else 1
        if drift:
            print(f"\n{drift} item(s) differ from PurpleLlama@{args.commit[:8]}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
