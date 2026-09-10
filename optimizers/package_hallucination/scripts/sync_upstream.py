#!/usr/bin/env python3
"""Verify the vendored package-hallucination data against garak.

``probes/packagehallucination.py`` cannot be imported (garak probe machinery),
so its three prompt/task lists and the ``language_name`` of each language
subclass are re-extracted here by AST and compared with ``data.json``. Nothing
fetched over the network is executed.

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
RAW = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/probes/packagehallucination.py"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src/package_hallucination_optimizer/data.json"
LISTS = ("stub_prompts", "real_code_tasks", "unreal_code_tasks")


def _extract(src: str) -> dict:
    t = ast.parse(src)
    out: dict = {}
    for n in t.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in LISTS:
            out[n.targets[0].id] = list(ast.literal_eval(n.value))
    langs = {}
    for n in ast.walk(t):
        if isinstance(n, ast.ClassDef):
            for s in n.body:
                if (
                    isinstance(s, ast.Assign)
                    and getattr(s.targets[0], "id", "") == "language_name"
                ):
                    langs[n.name] = ast.literal_eval(s.value)
    out["languages"] = langs
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    with urllib.request.urlopen(RAW.format(commit=args.commit)) as r:
        upstream = _extract(r.read().decode("utf-8"))

    keys = (*LISTS, "languages")
    missing = [k for k in keys if not upstream.get(k)]
    if missing:
        print(
            f"extraction failed: {missing} not found in garak@{args.commit[:8]}; "
            "upstream has probably moved. data.json is left untouched.",
            file=sys.stderr,
        )
        return 1

    local = json.loads(DATA.read_text("utf-8")) if DATA.exists() else {}
    drift = 0
    for k in keys:
        same = upstream.get(k) == local.get(k)
        print(f"{'ok  ' if same else 'DIFF'} {k}")
        drift += 0 if same else 1

    if not args.check and drift:
        DATA.write_text(
            json.dumps(upstream, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("updated data.json")
        return 0
    if args.check and drift:
        print(f"\n{drift} item(s) differ from garak@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
