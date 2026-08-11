"""The vendored upstream files must stay byte-identical to what was ported.

There is no public upstream commit to pin (the reference code is an anonymous,
under-review submission), so faithfulness is anchored to a SHA-256 of each
vendored file, recorded in ``constants.VENDORED_SHA256`` at vendoring time. This
test recomputes and compares, so any accidental edit under ``_vendor/`` fails
loudly. Every file the ``VENDORED_SHA256`` map names must exist, and every ``.py``
under ``_vendor/`` (except ``__init__``) must be covered by the map.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from attack_anything_optimizer.constants import VENDORED_SHA256

_VENDOR = Path(__file__).resolve().parent.parent / "src" / "attack_anything_optimizer" / "_vendor"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", sorted(VENDORED_SHA256))
def test_vendored_file_is_byte_identical(name: str) -> None:
    path = _VENDOR / name
    assert path.exists(), f"vendored file missing: {name}"
    assert _sha256(path) == VENDORED_SHA256[name], (
        f"{name} changed from the pinned upstream copy. _vendor/ is byte-identical "
        f"by design; if this is intentional, re-pin the SHA in constants.py."
    )


def test_every_vendored_module_is_pinned() -> None:
    on_disk = {p.name for p in _VENDOR.glob("*.py") if p.name != "__init__.py"}
    assert on_disk == set(VENDORED_SHA256), (
        "vendored modules and the SHA map have drifted:\n"
        f"  on disk but unpinned: {on_disk - set(VENDORED_SHA256)}\n"
        f"  pinned but missing:   {set(VENDORED_SHA256) - on_disk}"
    )
