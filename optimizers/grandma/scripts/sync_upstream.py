#!/usr/bin/env python3
"""Verify the vendored grandma data against garak.

``probes/grandma.py`` cannot be imported (it pulls in garak's probe machinery
and payload loader), so every list it carries is re-extracted here by AST and
compared with ``data.json``. The two payload files it loads are byte-compared
against garak's own copies. Nothing fetched over the network is executed.

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
RAW = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/{path}"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src/grandma_optimizer/data.json"
PAYLOADS = ROOT / "src/grandma_optimizer/_vendor/garak_payloads"
KEYS = (
    "PRODUCT_KEY_PROBE_TEMPLATES",
    "SLUR_PROBE_TEMPLATES",
    "SUBSTANCE_TEMPLATES",
    "human_groups",
    "substances",
    "actions",
    "personas",
    "person_adjs",
    "grandma_activities",
    "intent_template",
)


def _get(path: str, commit: str) -> str:
    with urllib.request.urlopen(RAW.format(commit=commit, path=path)) as r:
        return r.read().decode("utf-8")


def _extract(src: str) -> dict:
    """Every list grandma.py carries, wherever it carries it."""
    t = ast.parse(src)
    out: dict = {}
    for n in t.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "").endswith(
            "TEMPLATES"
        ):
            out[n.targets[0].id] = list(ast.literal_eval(n.value))
    for n in ast.walk(t):
        if isinstance(n, ast.ClassDef):
            for s in n.body:
                if (
                    isinstance(s, ast.Assign)
                    and getattr(s.targets[0], "id", "") == "DEFAULT_PARAMS"
                ):
                    for k, v in ast.literal_eval(s.value).items():
                        out[k] = list(v)
        if isinstance(n, ast.FunctionDef) and n.name == "_prompts_from_stub":
            for s in ast.walk(n):
                if not isinstance(s, ast.Assign):
                    continue
                name = getattr(s.targets[0], "id", "")
                if name in ("actions", "personas", "person_adjs", "grandma_activities"):
                    v = ast.literal_eval(s.value)
                    # grandma_activities is a set upstream; sort for a stable
                    # order (see ASSUMPTIONS "Upstream prompt order is not
                    # reproducible").
                    out[name] = sorted(v) if isinstance(v, set) else list(v)
                elif name == "template":
                    out["intent_template"] = ast.literal_eval(s.value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    upstream = _extract(_get("probes/grandma.py", args.commit))
    missing = [k for k in KEYS if k not in upstream]
    if missing:
        print(
            f"extraction failed: {missing} not found in garak@{args.commit[:8]}; "
            "upstream has probably moved. data.json is left untouched.",
            file=sys.stderr,
        )
        return 1

    local = json.loads(DATA.read_text("utf-8")) if DATA.exists() else {}
    drift = 0
    for k in KEYS:
        same = upstream.get(k) == local.get(k)
        print(f"{'ok  ' if same else 'DIFF'} {k}")
        drift += 0 if same else 1

    for name in ("keyedprod_win10", "keyedprod_win11"):
        up = _get(f"data/payloads/{name}.json", args.commit)
        dest = PAYLOADS / f"{name}.json"
        cur = dest.read_text(encoding="utf-8") if dest.exists() else ""
        same = cur == up
        print(f"{'ok  ' if same else 'DIFF'} {name}.json")
        if same:
            continue
        drift += 1
        if not args.check:
            dest.write_text(up, encoding="utf-8")
            print(f"updated   {name}.json")

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
