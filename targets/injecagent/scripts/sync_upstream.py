#!/usr/bin/env python3
"""Verify (or refresh) the vendored InjecAgent target data against upstream.

Downloads the InjecAgent repo at the pinned commit, then byte-compares the
vendored tool schemas and simulated-response cache against it.

    python scripts/sync_upstream.py            # refresh vendored files from upstream
    python scripts/sync_upstream.py --check     # verify byte-identical; exit 1 on drift

Nothing fetched over the network is executed.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import urllib.request

PINNED_COMMIT = "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"
REPO = "uiuc-kang-lab/InjecAgent"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "src", "injecagent_target", "data")

# vendored filename under _DATA  ->  path within the upstream tarball root
_FILES = [
    "tools.json",
    "attacker_simulated_responses.json",
]


def _pairs(root: str) -> list[tuple[str, str]]:
    return [(name, f"{root}/data/{name}") for name in _FILES]


def _download() -> tarfile.TarFile:
    url = f"https://github.com/{REPO}/archive/{PINNED_COMMIT}.tar.gz"
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - pinned https
        data = resp.read()
    return tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")


def main() -> int:
    check = "--check" in sys.argv
    tf = _download()
    root = tf.getnames()[0].split("/")[0]
    names = set(tf.getnames())
    drift = 0
    written = 0
    for rel, upstream_path in _pairs(root):
        if upstream_path not in names:
            # extractfile() raises KeyError (not None) for an absent name, so
            # membership must be checked first.
            print(f"MISSING upstream {upstream_path}")
            drift += 1
            continue
        member = tf.extractfile(upstream_path)
        up_bytes = member.read() if member else b""
        dest = os.path.join(_DATA, rel)
        cur = open(dest, "rb").read() if os.path.exists(dest) else None
        if cur == up_bytes:
            continue
        if check:
            print(f"DIFF {rel}")
            drift += 1
        else:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(up_bytes)
            written += 1
    if check:
        if drift:
            print(f"\n{drift} file(s) drifted from upstream {REPO}@{PINNED_COMMIT}")
            return 1
        print(f"ok: all vendored files byte-identical to {REPO}@{PINNED_COMMIT}")
        return 0
    print(f"refreshed {written} file(s) from {REPO}@{PINNED_COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
