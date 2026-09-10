#!/usr/bin/env python3
"""Verify the vendored apikey resources and prompts against garak.

Two garak resource files are vendored byte-identical and diffed:
``resources/apikey/regexes.py`` (the service list) and
``resources/apikey/serviceutils.py`` (``extract_key_types``). The two probe
classes' ``base_prompts`` and ``partial_keys`` are re-extracted by AST from
``probes/apikey.py`` -- which cannot be imported -- and compared with
``data.json``. Nothing fetched over the network is executed.

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
VENDOR = ROOT / "src/apikey_leak_optimizer/_vendor/garak_apikey"
DATA = ROOT / "src/apikey_leak_optimizer/data.json"


def _get(path: str, commit: str) -> str:
    with urllib.request.urlopen(RAW.format(commit=commit, path=path)) as r:
        return r.read().decode("utf-8")


def _extract(src: str) -> dict:
    t = ast.parse(src)
    out: dict = {}
    for n in ast.walk(t):
        if isinstance(n, ast.ClassDef) and n.name in ("GetKey", "CompleteKey"):
            cls = {}
            for s in ast.walk(n):
                if isinstance(s, ast.Assign):
                    tgt = s.targets[0]
                    name = getattr(tgt, "id", "") or getattr(tgt, "attr", "")
                    if name in ("base_prompts", "partial_keys"):
                        try:
                            cls[name] = ast.literal_eval(s.value)
                        except (ValueError, SyntaxError):
                            pass
            out[n.name] = cls
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    drift = 0
    for name in ("regexes.py", "serviceutils.py"):
        up = _get(f"resources/apikey/{name}", args.commit)
        dest = VENDOR / name
        cur = dest.read_text(encoding="utf-8") if dest.exists() else ""
        same = cur == up
        print(f"{'ok  ' if same else 'DIFF'} {name}")
        if same:
            continue
        drift += 1
        if not args.check:
            dest.write_text(up, encoding="utf-8")
            print(f"updated   {name}")

    upstream = _extract(_get("probes/apikey.py", args.commit))
    if not (upstream.get("GetKey") and upstream.get("CompleteKey")):
        print(
            f"extraction failed: probe constants not found in garak@{args.commit[:8]}; "
            "upstream has probably moved. data.json is left untouched.",
            file=sys.stderr,
        )
        return 1

    local = json.loads(DATA.read_text("utf-8")) if DATA.exists() else {}
    same = upstream == local
    print(f"{'ok  ' if same else 'DIFF'} probe prompts")
    if not same:
        drift += 1
        if not args.check:
            DATA.write_text(
                json.dumps(upstream, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print("updated data.json")

    if args.check and drift:
        print(f"\n{drift} item(s) differ from garak@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
