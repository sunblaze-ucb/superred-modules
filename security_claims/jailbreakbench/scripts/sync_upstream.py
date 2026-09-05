#!/usr/bin/env python3
"""Verify/refresh the vendored JailbreakBench data and judge prompt.

    python scripts/sync_upstream.py [--commit SHA] [--check]

Checks two things against upstream: the harmful-behaviors CSV (from the
JBB-Behaviors HF dataset) and the judge USER_PROMPT (from the classifier).
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "23dbdf6b19650521604456229bc1d9c4156c85c1"
CSV_URL = "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv"
CLASSIFIER_URL = "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/{commit}/src/jailbreakbench/classifier.py"

ROOT = Path(__file__).resolve().parent.parent
CSV_DEST = ROOT / "src/jailbreakbench_claim/data/harmful-behaviors.csv"
JUDGE_SRC = ROOT / "src/jailbreakbench_claim/judge.py"


def _get(url: str) -> str:
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def _upstream_prompt(commit: str) -> str:
    src = _get(CLASSIFIER_URL.format(commit=commit))
    m = re.search(
        r'class Llama3JailbreakJudge\(Llama3Classifier\):\s*USER_PROMPT = (""".*?""")',
        src,
        re.S,
    )
    if not m:
        raise SystemExit("could not find Llama3JailbreakJudge.USER_PROMPT upstream")
    return m.group(1)[3:-3]  # strip the triple quotes


def _local_prompt() -> str:
    src = JUDGE_SRC.read_text(encoding="utf-8")
    m = re.search(r'USER_PROMPT = (""".*?""")', src, re.S)
    return m.group(1)[3:-3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    csv_up = _get(CSV_URL)
    csv_local = CSV_DEST.read_text(encoding="utf-8") if CSV_DEST.exists() else ""
    # \\ USER_PROMPT uses a leading-backslash line continuation, so upstream's
    # literal begins with "\\\n"; strip that when comparing the rendered text.
    prompt_up = _upstream_prompt(args.commit).lstrip("\\\n")
    prompt_local = _local_prompt().lstrip("\\\n")

    csv_same = csv_up == csv_local
    prompt_same = prompt_up == prompt_local

    if args.check:
        print(f"{'ok  ' if csv_same else 'DIFF'} harmful-behaviors.csv")
        print(f"{'ok  ' if prompt_same else 'DIFF'} judge USER_PROMPT")
        if not (csv_same and prompt_same):
            print("drift from upstream", file=sys.stderr)
            return 1
        return 0

    if not csv_same:
        CSV_DEST.write_text(csv_up, encoding="utf-8")
        print("updated   harmful-behaviors.csv")
    else:
        print("unchanged harmful-behaviors.csv")
    if not prompt_same:
        print("NOTE: judge USER_PROMPT differs; edit judge.py by hand", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
