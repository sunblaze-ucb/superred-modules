"""Fixture loading helpers for the executable subset."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Scenario


DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "tiny_subset.json"


def load_fixture_scenarios(
    fixture_path: str | Path = DEFAULT_FIXTURE_PATH,
    limit: int | None = None,
) -> list[Scenario]:
    path = Path(fixture_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("fixture payload must be a list of scenario objects")
    scenarios = [Scenario.from_dict(item) for item in payload]
    return scenarios[:limit] if limit is not None else scenarios
