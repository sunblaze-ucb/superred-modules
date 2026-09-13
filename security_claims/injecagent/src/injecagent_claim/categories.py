"""Corpus loader for the vendored InjecAgent test cases.

Two attack types (``dh`` = direct harm, ``ds`` = data stealing) times two
settings (``base``, ``enhanced`` = injection reinforced with a hacking prompt),
one JSON file each, all vendored byte-for-byte from upstream ``f19c9f2``.

Upstream ships no per-case identifier, so a stable ``task_id`` is synthesized
from ``{attack}_{setting}_{index}`` — deterministic because the corpus is pinned
and vendored verbatim (index = position in the upstream file).
"""

from __future__ import annotations

import json
import os
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")

ATTACKS: tuple[str, ...] = ("dh", "ds")
SETTINGS: tuple[str, ...] = ("base", "enhanced")


def _load_file(attack: str, setting: str) -> list[dict[str, Any]]:
    path = os.path.join(DATA_DIR, f"test_cases_{attack}_{setting}.json")
    with open(path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def load_cases(
    settings: tuple[str, ...] | list[str] = ("base",),
    attacks: tuple[str, ...] | list[str] = ATTACKS,
) -> list[dict[str, Any]]:
    """Return the requested test cases, each augmented with routing metadata.

    Each returned dict is a copy of the verbatim upstream case plus three keys:
    ``_attack`` (dh/ds), ``_setting`` (base/enhanced), and ``_task_id``.
    Ordered by (setting, attack, upstream index) for a deterministic corpus.
    """
    for s in settings:
        if s not in SETTINGS:
            raise ValueError(f"unknown setting {s!r}; expected one of {SETTINGS}")
    for a in attacks:
        if a not in ATTACKS:
            raise ValueError(f"unknown attack type {a!r}; expected one of {ATTACKS}")

    out: list[dict[str, Any]] = []
    for setting in settings:
        for attack in attacks:
            for idx, case in enumerate(_load_file(attack, setting)):
                c = dict(case)
                c["_attack"] = attack
                c["_setting"] = setting
                c["_task_id"] = f"{attack}_{setting}_{idx:03d}"
                out.append(c)
    return out


def task_counts() -> dict[str, int]:
    """Per-file case counts (``{attack}_{setting} -> n``)."""
    return {f"{a}_{s}": len(_load_file(a, s)) for s in SETTINGS for a in ATTACKS}


__all__ = ["ATTACKS", "SETTINGS", "DATA_DIR", "load_cases", "task_counts"]
