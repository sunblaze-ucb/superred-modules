"""The three judges must fail identically, so their policy must be identical.

``judge_failure.py`` is duplicated verbatim into each of the three
chatbot benchmark claim packages, because each is an independently
installable distribution with nowhere shared to put it. Duplication
without a drift check is how three judges end up with three policies
again, which is the defect this module exists to remove.

Skipped when the sibling packages are not on disk (e.g. running against
an installed wheel rather than the repository checkout).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _sibling_copies() -> list[Path]:
    security_claims = Path(__file__).resolve().parents[2]
    return sorted(security_claims.glob("*/src/*/judge_failure.py"))


def test_all_copies_are_byte_identical() -> None:
    copies = _sibling_copies()
    if len(copies) < 2:
        pytest.skip(f"sibling claim packages not on disk (found {len(copies)})")
    contents = {path: path.read_bytes() for path in copies}
    reference_path, reference = next(iter(contents.items()))
    drifted = [
        str(path) for path, data in contents.items() if data != reference
    ]
    assert not drifted, (
        f"judge_failure.py has drifted from {reference_path} in: {drifted}. "
        "Edit all copies together — the three judges are required to fail "
        "the same way."
    )
