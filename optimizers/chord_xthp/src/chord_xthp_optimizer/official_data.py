"""Load official Chord/XTHP data artifacts used by the optimizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Literal, cast

Direction = Literal["predecessor", "successor"]
AttackName = Literal["hijack", "harvest", "pollute"]
JsonObject = dict[str, Any]

_OFFICIAL_PACKAGE = "chord_xthp_optimizer"
_OFFICIAL_DIR = ("data", "official")


@dataclass(frozen=True)
class ChordToolInfo:
    name: str
    description: str


def _load_json(name: str) -> Any:
    path = resources.files(_OFFICIAL_PACKAGE).joinpath(*_OFFICIAL_DIR, name)
    return json.loads(path.read_text(encoding="utf-8"))


def _direction_items(data: JsonObject, direction: Direction) -> list[JsonObject]:
    langchain = data.get("langchain")
    if not isinstance(langchain, dict):
        raise TypeError("official Chord malicious tool data must contain langchain object")
    rows = langchain.get(direction)
    if not isinstance(rows, list):
        raise TypeError(f"official Chord malicious tool data missing {direction} list")
    return [item for item in rows if isinstance(item, dict)]


@lru_cache(maxsize=2)
def load_official_malicious_tools(direction: Direction) -> dict[str, ChordToolInfo]:
    """Return victim-tool -> malicious helper metadata for one Chord direction."""

    data = _load_json("malicious_tools.json")
    if not isinstance(data, dict):
        raise TypeError("official Chord malicious_tools.json must be an object")
    out: dict[str, ChordToolInfo] = {}
    for row in _direction_items(cast(JsonObject, data), direction):
        for victim, info in row.items():
            if not isinstance(victim, str) or not isinstance(info, dict):
                continue
            name = info.get("name")
            description = info.get("description")
            if not isinstance(name, str) or not isinstance(description, str):
                raise TypeError("official Chord malicious tool rows need name/description")
            out[victim] = ChordToolInfo(name=name, description=description)
    return out


@lru_cache(maxsize=2)
def load_official_malicious_tool_arguments(direction: Direction) -> dict[str, dict[str, list[str]]]:
    """Return Chord's sensitive-data labels and helper parameter names."""

    data = _load_json("malicious_tool_arguments.json")
    if not isinstance(data, dict):
        raise TypeError("official Chord malicious_tool_arguments.json must be an object")
    rows = data.get(direction)
    if not isinstance(rows, dict):
        raise TypeError(f"official Chord malicious tool arguments missing {direction}")
    out: dict[str, dict[str, list[str]]] = {}
    for victim, mapping in rows.items():
        if not isinstance(victim, str) or not isinstance(mapping, dict):
            continue
        victim_mapping: dict[str, list[str]] = {}
        for sensitive_label, params in mapping.items():
            if not isinstance(sensitive_label, str) or not isinstance(params, list):
                continue
            victim_mapping[sensitive_label] = [str(param) for param in params]
        out[victim] = victim_mapping
    return out


@lru_cache(maxsize=1)
def load_official_queries() -> dict[str, tuple[str, ...]]:
    """Return Chord's official query set keyed by victim tool name."""

    data = _load_json("query.json")
    if not isinstance(data, dict):
        raise TypeError("official Chord query.json must be an object")
    out: dict[str, tuple[str, ...]] = {}
    for victim, queries in data.items():
        if isinstance(victim, str) and isinstance(queries, list):
            out[victim] = tuple(str(query) for query in queries if isinstance(query, str))
    return out


__all__ = [
    "AttackName",
    "ChordToolInfo",
    "Direction",
    "load_official_malicious_tool_arguments",
    "load_official_malicious_tools",
    "load_official_queries",
]
