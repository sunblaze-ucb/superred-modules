#!/usr/bin/env python3
"""Verify the vendored multilingual template + compliance prompt against DeepTeam.

template.py is a byte-identical copy. compliance.py cannot be vendored whole
(it imports DeepTeam's generate helper), so build_compliance_check_prompt is
reproduced here and checked by re-extracting the upstream function and
comparing the prompt it produces.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import ast
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "dc148aad62f71330cfec7121d6afb4c620dfa683"
BASE = "https://raw.githubusercontent.com/confident-ai/deepteam/{commit}/{path}"
TEMPLATE_PATH = "deepteam/attacks/single_turn/multilingual/template.py"
COMPLIANCE_PATH = "deepteam/attacks/single_turn/compliance.py"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "src/multilingual_optimizer/_vendor/dt_multilingual/template.py"


def _get(path: str, commit: str) -> str:
    with urllib.request.urlopen(BASE.format(commit=commit, path=path)) as r:
        return r.read().decode("utf-8")


def _extract_fn(src: str, name: str):
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            g: dict = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<up>", "exec"), g)
            return g[name]
    raise SystemExit(f"could not find {name} upstream")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    drift = 0
    # 1. template.py byte-identical
    up_tpl = _get(TEMPLATE_PATH, args.commit)
    local = DEST.read_text(encoding="utf-8") if DEST.exists() else ""
    same = local == up_tpl
    if args.check:
        print(f"{'ok  ' if same else 'DIFF'} template.py")
        drift += 0 if same else 1
    elif not same:
        DEST.write_text(up_tpl, encoding="utf-8")
        print("updated   template.py")
    else:
        print("unchanged template.py")

    # 2. compliance prompt reproduced faithfully
    sys.path.insert(0, str(ROOT / "src"))
    from multilingual_optimizer.compliance import build_compliance_check_prompt as mine

    up_fn = _extract_fn(_get(COMPLIANCE_PATH, args.commit), "build_compliance_check_prompt")
    prompt_same = mine("PROBE") == up_fn("PROBE")
    print(f"{'ok  ' if prompt_same else 'DIFF'} compliance prompt")
    drift += 0 if prompt_same else 1

    if args.check and drift:
        print(f"\n{drift} item(s) differ from deepteam@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
