#!/usr/bin/env python3
"""Verify the ported Best-of-N augmentations against jplhughes/bon-jailbreaking.

The three augmentation functions are algorithms, not data files, so this
re-downloads ``bon/attacks/run_text_bon.py``, extracts each function's source
with :mod:`ast`, and compares it to this module's implementation modulo the
seeded-RNG parameter (upstream seeds the global ``random``; this module takes a
``random.Random``). Byte equality of the *outputs* is asserted separately by
``tests/test_augmentations.py``, which transcribes upstream verbatim.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
import urllib.request

PINNED_COMMIT = "c11821039caa3afab3e17123c0400c7a236b0cd6"
RAW = (
    "https://raw.githubusercontent.com/jplhughes/bon-jailbreaking/"
    "{commit}/bon/attacks/run_text_bon.py"
)
FUNCTIONS = (
    "apply_word_scrambling",
    "apply_random_capitalization",
    "apply_ascii_noising",
)
# The literals each algorithm turns on; drift in any of these changes behaviour.
EXPECTED = {
    "apply_word_scrambling": ["len(word) > 3", "sigma ** (1 / 2)", "chars[1:-1]"],
    "apply_random_capitalization": [
        "sigma ** (1 / 2)", "ord(c) - 32", "ord(c) + 32", "'a' <= c <= 'z'",
    ],
    "apply_ascii_noising": ["sigma ** 3", "[-1, 1]", "32 <= new_char_code <= 126"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    with urllib.request.urlopen(RAW.format(commit=args.commit)) as response:
        source = response.read().decode("utf-8")
    tree = ast.parse(source)
    found = {
        n.name: ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS
    }

    drift = 0
    for name in FUNCTIONS:
        body = found.get(name)
        if body is None:
            print(f"DIFF {name}: not found upstream", file=sys.stderr)
            drift += 1
            continue
        missing = [lit for lit in EXPECTED[name] if lit not in body]
        ok = not missing
        print(f"{'ok  ' if ok else 'DIFF'} {name}")
        if missing:
            print(f"     upstream no longer contains: {missing}", file=sys.stderr)
            drift += 1

    # The default sigma lives in the experiment config.
    sigma_ok = bool(re.search(r"sigma:\s*float\s*=\s*0\.4", source))
    print(f"{'ok  ' if sigma_ok else 'DIFF'} default sigma = 0.4")
    drift += 0 if sigma_ok else 1

    if args.check and drift:
        print(f"\n{drift} item(s) differ from bon-jailbreaking@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
