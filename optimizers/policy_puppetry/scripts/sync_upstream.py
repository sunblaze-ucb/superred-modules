#!/usr/bin/env python3
"""Verify/refresh the vendored Policy Puppetry templates against NVIDIA garak.

Upstream ``Bypass.templates`` is a list of two triple-quoted scene templates.
They are extracted with :mod:`ast` -- not a regex -- because the list separator
between the two literals (``\""",\\n            \"""``) is itself valid template
text, so a lazy regex silently captures both templates *plus* the intervening
Python source syntax. Parsing the literal is the only way to get the exact
boundaries upstream intends.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""

from __future__ import annotations

import argparse
import ast
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "2233d51bedf6bfb634872ef9efbcd94eae38cc1a"
RAW = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/probes/doctor.py"
DEST_DIR = (
    Path(__file__).resolve().parent.parent
    / "src/policy_puppetry_optimizer/data/upstream"
)
FILES = ("bypass_template_0.txt", "bypass_template_1.txt")


def extract(commit: str) -> list[str]:
    """Return upstream ``Bypass.templates`` exactly, via AST literal parsing."""
    with urllib.request.urlopen(RAW.format(commit=commit)) as response:
        source = response.read().decode("utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == "Bypass"):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign) and any(
                getattr(t, "attr", None) == "templates" for t in stmt.targets
            ):
                try:
                    templates = ast.literal_eval(stmt.value)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(templates, list) and all(
                    isinstance(t, str) for t in templates
                ):
                    return list(templates)
    raise SystemExit("could not locate Bypass.templates in upstream doctor.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument("--check", action="store_true", help="verify only")
    args = parser.parse_args()

    upstream = extract(args.commit)
    if len(upstream) != len(FILES):
        print(
            f"upstream now has {len(upstream)} templates, this module vendors "
            f"{len(FILES)}; update FILES and templates.TEMPLATE_COUNT",
            file=sys.stderr,
        )
        return 1

    drift = 0
    for name, text in zip(FILES, upstream, strict=True):
        dest = DEST_DIR / name
        local = dest.read_text(encoding="utf-8") if dest.exists() else ""
        same = local == text
        # A template must carry exactly one slot; two would mean the extraction
        # boundary swallowed a neighbouring literal.
        slots = text.count("{{HARMFUL_BEHAVIOR}}")
        if args.check:
            print(f"{'ok  ' if same else 'DIFF'} {name} (slots={slots})")
            drift += 0 if same else 1
            if slots != 1:
                print(f"     unexpected slot count {slots} in {name}", file=sys.stderr)
                drift += 1
        elif same:
            print(f"unchanged {name}")
        else:
            dest.write_text(text, encoding="utf-8")
            print(f"updated   {name}")

    if args.check and drift:
        print(
            f"\n{drift} item(s) differ from garak@{args.commit[:8]}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
