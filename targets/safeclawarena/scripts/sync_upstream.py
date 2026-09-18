#!/usr/bin/env python3
"""Verify (or refresh) the vendored SafeClawArena container harness against upstream.

Downloads the SafeClawArena repo at the pinned commit, then byte-compares every
vendored harness file (Dockerfiles, sim-google CLI, reset_env.sh,
configs) against it.

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
_VENDOR = os.path.join(_HERE, "..", "src", "safeclawarena_target", "_vendor", "safeclawarena")


def _vendored_pairs(root: str) -> list[tuple[str, str]]:
    """(vendored path under _VENDOR) -> (path within the upstream tarball).

    Enumerates every vendored file EXCEPT this module's own added LICENSE copy.
    """
    pairs: list[tuple[str, str]] = []
    for dirpath, _dirs, files in os.walk(_VENDOR):
        for name in files:
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, _VENDOR)
            if rel == "LICENSE":
                # the module ships upstream's LICENSE at the vendor root; map it
                pairs.append((rel, f"{root}/LICENSE"))
            else:
                pairs.append((rel, f"{root}/{rel}"))
    return sorted(pairs)


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
    for rel, upstream_path in _vendored_pairs(root):
        if upstream_path not in names:
            print(f"MISSING upstream {upstream_path}")
            drift += 1
            continue
        member = tf.extractfile(upstream_path)
        up_bytes = member.read() if member else b""
        dest = os.path.join(_VENDOR, rel)
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
        print(f"ok: all vendored harness files byte-identical to {REPO}@{PINNED_COMMIT}")
        return 0
    print(f"refreshed {written} file(s) from {REPO}@{PINNED_COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
