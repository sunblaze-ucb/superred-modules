"""Packaged official DRA data files."""

from __future__ import annotations

import json
from importlib import resources

_BEHAVIORS_RESOURCE = "behaviors.json"


def behaviors_json_bytes() -> bytes:
    """Return the official DRA behavior file exactly as packaged."""

    return resources.files(__name__).joinpath(_BEHAVIORS_RESOURCE).read_bytes()


def load_behaviors() -> list[str]:
    """Load the official DRA `data/behaviors.json` behavior list."""

    raw = json.loads(behaviors_json_bytes().decode("utf-8"))
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("DRA behaviors.json must contain a JSON list of strings")
    return raw


__all__ = ["behaviors_json_bytes", "load_behaviors"]
