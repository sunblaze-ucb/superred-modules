#!/usr/bin/env python3
"""Verify the vendored DeepInception template against the upstream README.

The template lives in the README's "Ready-to-Use Prompt" as a worked instance
(scene=dream, 5 characters, 5 layers, a sample query). This renders our template
with those exact values and compares to the README instance, byte-for-byte.

    python scripts/sync_upstream.py [--commit SHA] [--check]
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request

PINNED_COMMIT = "fc5689e76c4dd87a14babf1eadf890532f2b6880"
README = "https://raw.githubusercontent.com/tmlr-group/DeepInception/{commit}/README.md"
SAMPLE_QUERY = "show how to hack into a Linux computer"


def _readme_instance(commit: str) -> str:
    with urllib.request.urlopen(README.format(commit=commit)) as r:
        md = r.read().decode("utf-8")
    # The instance is the paragraph starting "Create a" with <b> bold tags.
    block = re.search(r"(Create a .*?each layer\.)", md, re.S)
    if not block:
        raise SystemExit("could not find the README template instance")
    return re.sub(r"</?b>", "", block.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    import sys as _sys
    from pathlib import Path

    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from deepinception_optimizer.template import render

    ours = render(SAMPLE_QUERY)
    upstream = _readme_instance(args.commit)
    same = ours == upstream
    print(f"{'ok  ' if same else 'DIFF'} DeepInception template vs README instance")
    if not same:
        for i, (a, b) in enumerate(zip(ours, upstream)):
            if a != b:
                print(f"  first diff @ {i}: {ours[i-20:i+20]!r} vs {upstream[i-20:i+20]!r}", file=sys.stderr)
                break
        print(f"  lens: ours={len(ours)} readme={len(upstream)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
