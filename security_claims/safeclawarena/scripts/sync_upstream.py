#!/usr/bin/env python3
"""Verify (or refresh) the vendored SafeClawArena benchmark data against upstream.

Downloads the SafeClawArena repo at the pinned commit, then byte-compares every
vendored task JSON + the task schema against it.

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

PINNED_COMMIT = "a11f5cc"
REPO = "sunblaze-ucb/SafeClawArena"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "src", "safeclawarena_claim", "data")

# (vendored relative path under _DATA)  ->  (path within the upstream tarball root)
def _pairs(root: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = [("task_schema.json", f"{root}/schema/task_schema.json")]
    tasks_dir = os.path.join(_DATA, "tasks")
    for dim in sorted(os.listdir(tasks_dir)):
        dim_dir = os.path.join(tasks_dir, dim)
        if not os.path.isdir(dim_dir):
            continue
        for name in sorted(os.listdir(dim_dir)):
            if name.endswith(".json"):
                pairs.append((f"tasks/{dim}/{name}", f"{root}/tasks/{dim}/{name}"))
    return pairs


def _download() -> tarfile.TarFile:
    url = f"https://github.com/{REPO}/archive/{PINNED_COMMIT}.tar.gz"
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - pinned https
        data = resp.read()
    return tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")


def main() -> int:
    check = "--check" in sys.argv
    tf = _download()
    root = tf.getnames()[0].split("/")[0]
    drift = 0
    written = 0
    for rel, upstream_path in _pairs(root):
        member = tf.extractfile(upstream_path)
        if member is None:
            print(f"MISSING upstream {upstream_path}")
            drift += 1
            continue
        up_bytes = member.read()
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
