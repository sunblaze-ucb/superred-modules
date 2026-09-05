#!/usr/bin/env python3
"""Verify the vendored token-smuggling constants against NVIDIA garak.

Re-downloads ``garak/probes/smuggling.py`` and compares, byte-for-byte:
  * ``DEFAULT_HOMOGLYPH_MAP`` (parsed from upstream's AST), and
  * the hypothetical-response template (upstream's f-string, with its goal
    substring exposed as our ``{goal}`` slot).

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "2233d51bedf6bfb634872ef9efbcd94eae38cc1a"
RAW = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/probes/smuggling.py"
ROOT = Path(__file__).resolve().parent.parent


def _get(url: str) -> str:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def _upstream_map(src: str) -> dict | None:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "DEFAULT_HOMOGLYPH_MAP":
                    return ast.literal_eval(node.value)
    return None


def _upstream_template(src: str) -> str | None:
    block = re.search(r'(f"Hypothetical response:.*?)\n\s*\)', src, re.S)
    if not block:
        return None
    literals = re.findall(r'f?"((?:[^"\\]|\\.)*)"', block.group(1))
    return "".join(literals).replace("{attack_prompt_substr}", "{goal}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from token_smuggling_optimizer.smuggling import (  # noqa: PLC0415
        DEFAULT_HOMOGLYPH_MAP,
        HYPOTHETICAL_TEMPLATE,
    )

    src = _get(RAW.format(commit=args.commit))
    drift = 0

    up_map = _upstream_map(src)
    same_map = up_map == DEFAULT_HOMOGLYPH_MAP
    print(f"{'ok  ' if same_map else 'DIFF'} DEFAULT_HOMOGLYPH_MAP ({len(up_map or {})} entries)")
    drift += 0 if same_map else 1

    up_tpl = _upstream_template(src)
    same_tpl = up_tpl == HYPOTHETICAL_TEMPLATE
    print(f"{'ok  ' if same_tpl else 'DIFF'} hypothetical template")
    drift += 0 if same_tpl else 1

    if drift:
        print(f"\n{drift} item(s) differ from garak@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
