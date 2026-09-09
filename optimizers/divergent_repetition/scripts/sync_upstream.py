#!/usr/bin/env python3
"""Verify data.json against garak's divergence.Repeat / RepeatExtended.

The word lists, prompt templates and repetition counts are data extracted from
garak/probes/divergence.py. This re-downloads that probe, re-extracts the same
literals with ast, and compares them to data.json.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "2233d51bedf6bfb634872ef9efbcd94eae38cc1a"
RAW = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/probes/divergence.py"
DEST = Path(__file__).resolve().parent.parent / "src/divergent_repetition_optimizer/data.json"


def _extract(src: str) -> dict:
    tree = ast.parse(src)
    out: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Repeat":
            for s in node.body:
                if isinstance(s, ast.Assign) and getattr(s.targets[0], "id", "") == "repeat_word_list":
                    out["repeat_words"] = ast.literal_eval(s.value)
                if isinstance(s, ast.FunctionDef) and s.name == "__init__":
                    for st in ast.walk(s):
                        if isinstance(st, ast.Assign):
                            nm = getattr(st.targets[0], "id", "")
                            if nm == "prompt_templates":
                                out["prompt_templates"] = ast.literal_eval(st.value)
                            if nm == "num_repetitions":
                                out["num_repetitions"] = ast.literal_eval(st.value)
        if isinstance(node, ast.ClassDef) and node.name == "RepeatExtended":
            for s in node.body:
                if isinstance(s, ast.Assign) and getattr(s.targets[0], "id", "") == "repeat_word_list":
                    out["repeat_words_extended"] = ast.literal_eval(s.value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    with urllib.request.urlopen(RAW.format(commit=args.commit)) as r:
        upstream = _extract(r.read().decode("utf-8"))
    local = json.loads(DEST.read_text("utf-8")) if DEST.exists() else {}

    keys = ("repeat_words", "prompt_templates", "num_repetitions", "repeat_words_extended")

    # A rename or a moved assignment upstream makes _extract return fewer keys.
    # Without this, every missing key reads as ordinary drift and the default
    # (non---check) mode writes the truncated dict straight over data.json.
    missing = [k for k in keys if k not in upstream]
    if missing:
        print(
            f"extraction failed: {missing} not found in garak@{args.commit[:8]}; "
            "upstream has probably moved. data.json is left untouched.",
            file=sys.stderr,
        )
        return 1

    drift = 0
    for k in keys:
        same = upstream.get(k) == local.get(k)
        print(f"{'ok  ' if same else 'DIFF'} {k}")
        drift += 0 if same else 1

    if not args.check and drift:
        DEST.write_text(json.dumps(upstream, indent=1, ensure_ascii=False) + "\n", "utf-8")
        print("updated data.json")
        return 0
    if args.check and drift:
        print(f"\n{drift} item(s) differ from garak@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
