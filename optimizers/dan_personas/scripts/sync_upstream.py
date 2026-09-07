#!/usr/bin/env python3
"""Refresh the vendored DAN persona files from NVIDIA garak.

Downloads ``garak/data/dan/<name>.json`` at a pinned commit and writes it
byte-identically into ``src/dan_personas_optimizer/data/upstream/``. Run this
to move to a newer upstream, then update PINNED_COMMIT, NOTICE and
LICENSES/NOTICE.md together so the recorded provenance stays true.

    python scripts/sync_upstream.py [--commit SHA] [--check]

``--check`` verifies the vendored files match upstream without writing.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

PINNED_COMMIT = "2233d51bedf6bfb634872ef9efbcd94eae38cc1a"
RAW = "https://raw.githubusercontent.com/NVIDIA/garak/{commit}/garak/data/dan/{name}.json"

NAMES = (
    "AntiDAN",
    "ChatGPT_Developer_Mode_RANTI",
    "ChatGPT_Developer_Mode_v2",
    "ChatGPT_Image_Markdown",
    "DAN_Jailbreak",
    "DUDE",
    "Dan_10_0",
    "Dan_11_0",
    "Dan_6_0",
    "Dan_6_2",
    "Dan_7_0",
    "Dan_8_0",
    "Dan_9_0",
    "STAN",
)

DEST = Path(__file__).resolve().parent.parent / (
    "src/dan_personas_optimizer/data/upstream"
)


def fetch(name: str, commit: str) -> bytes:
    with urllib.request.urlopen(RAW.format(commit=commit, name=name)) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument(
        "--check", action="store_true", help="verify only; write nothing"
    )
    args = parser.parse_args()

    drift = 0
    # DanInTheWild's corpus lives outside garak/data/dan/.
    extra = {"inthewild_jailbreak_llms.json": "garak/data/inthewild_jailbreak_llms.json"}
    for local, remote in extra.items():
        url = f"https://raw.githubusercontent.com/NVIDIA/garak/{args.commit}/{remote}"
        with urllib.request.urlopen(url) as response:
            up = response.read()
        dest = DEST / local
        same = dest.exists() and dest.read_bytes() == up
        if args.check:
            print(f"{'ok  ' if same else 'DIFF'} {local}")
            drift += 0 if same else 1
        elif not same:
            dest.write_bytes(up)
            print(f"updated   {local}")

    for name in NAMES:
        upstream = fetch(name, args.commit)
        target = DEST / f"{name}.json"
        local = target.read_bytes() if target.exists() else b""
        same = hashlib.sha256(local).digest() == hashlib.sha256(upstream).digest()
        if args.check:
            print(f"{'ok  ' if same else 'DIFF'} {name}")
            drift += 0 if same else 1
        elif same:
            print(f"unchanged {name}")
        else:
            target.write_bytes(upstream)
            print(f"updated   {name}")

    if args.check and drift:
        print(f"\n{drift} file(s) differ from garak@{args.commit[:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
