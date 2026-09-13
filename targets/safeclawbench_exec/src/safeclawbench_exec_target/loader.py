"""Load the vendored SafeClawBench Exec-Balanced scenarios.

The 600-case executable set (``exec_full_600.json``) and a ``tiny_subset`` are
vendored verbatim under ``_vendor/executable/fixtures/``; this thin loader
parses them into the vendored :class:`Scenario` dataclass and exposes the six
reporting families.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from safeclawbench_exec_target._vendor.executable.schema import Scenario

_FIXTURES: Final = (
    Path(__file__).resolve().parent / "_vendor" / "executable" / "fixtures"
)
EXEC_FULL_PATH: Final = _FIXTURES / "exec_full_600.json"
TINY_SUBSET_PATH: Final = _FIXTURES / "tiny_subset.json"

#: The six SafeClawBench reporting families (100 exec cases each in the full set).
ATTACK_FAMILIES: Final = ("DPI", "IPI", "TRI", "MPI", "MEX", "ADI")

FAMILY_NAMES: Final = {
    "DPI": "direct prompt injection",
    "IPI": "indirect prompt injection",
    "TRI": "tool-return injection",
    "MPI": "memory poisoning",
    "MEX": "memory extraction",
    "ADI": "ambiguity-driven unsafe inference",
}


def family_label(attack_type: str) -> str:
    """Human-readable label for a reporting family code (e.g. ``"DPI"``)."""
    return FAMILY_NAMES.get(attack_type, attack_type)


def load_scenarios(path: str | Path = EXEC_FULL_PATH) -> list[Scenario]:
    """Load and parse all executable scenarios from a vendored fixture file."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("exec fixture payload must be a JSON list of scenarios")
    return [Scenario.from_dict(item) for item in payload]


__all__ = [
    "ATTACK_FAMILIES",
    "EXEC_FULL_PATH",
    "FAMILY_NAMES",
    "TINY_SUBSET_PATH",
    "family_label",
    "load_scenarios",
]
