#!/usr/bin/env python3
"""Verify the vendored Prefill template against Tencent AI-Infra-Guard.

The template is a byte-identical copy of upstream's
``single_turn/prefill/template.py``; this re-downloads it and diffs. The
one-line ``enhance`` is reproduced in ``optimizer.render``, so its source is
compared too -- without executing anything fetched over the network.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""

from __future__ import annotations

import argparse
import ast
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "dd6bd54655c9ff5fb7351f4299b56916f09ec6da"
BASE = (
    "https://raw.githubusercontent.com/Tencent/AI-Infra-Guard/{commit}/"
    "AIG-PromptSecurity/deepteam/attacks/single_turn/prefill/{name}.py"
)
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "src/prefill_optimizer/_vendor/aig_prefill/template.py"


def _get(name: str, commit: str) -> str:
    with urllib.request.urlopen(BASE.format(commit=commit, name=name)) as r:
        return r.read().decode("utf-8")


def _joiner(src: str, fn_name: str) -> list[str]:
    """The literal parts of the f-string ``fn_name`` returns, executing nothing.

    ``f"{attack}\n\n{self.prefill}"`` -> ``["", "\n\n", ""]``. Comparing the
    literals pins the separator exactly while ignoring what the interpolated
    names happen to be called on each side.
    """
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            for st in ast.walk(node):
                if isinstance(st, ast.JoinedStr):
                    return [
                        v.value for v in st.values if isinstance(v, ast.Constant)
                    ]
    raise SystemExit(f"could not find an f-string in {fn_name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    drift = 0

    upstream = _get("template", args.commit)
    current = DEST.read_text(encoding="utf-8") if DEST.exists() else ""
    same = current == upstream
    if args.check:
        print(f"{'ok  ' if same else 'DIFF'} template.py")
        drift += 0 if same else 1
    elif not same:
        DEST.write_text(upstream, encoding="utf-8")
        print("updated   template.py")
    else:
        print("unchanged template.py")

    # enhance is one f-string; compare its literal parts, not its variable names.
    up_parts = _joiner(_get("prefill", args.commit), "enhance")
    our_parts = _joiner(
        (ROOT / "src/prefill_optimizer/optimizer.py").read_text(encoding="utf-8"),
        "render",
    )
    body_same = up_parts == our_parts
    print(f"{'ok  ' if body_same else 'DIFF'} enhance separator {up_parts!r}")
    drift += 0 if body_same else 1

    if args.check and drift:
        print(f"\n{drift} item(s) differ from AI-Infra-Guard@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
