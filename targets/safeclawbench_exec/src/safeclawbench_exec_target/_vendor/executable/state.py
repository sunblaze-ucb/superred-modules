"""Mock world state for executable SafeClawBench cases."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class MockWorldState:
    """State namespaces exposed to the mock tools."""

    files: dict[str, str] = field(default_factory=dict)
    email: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {"sent": [], "inbox": []}
    )
    db: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    web: dict[str, Any] = field(default_factory=dict)
    code: dict[str, Any] = field(default_factory=lambda: {"executions": []})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MockWorldState":
        data = data or {}
        email = copy.deepcopy(data.get("email", {"sent": [], "inbox": []}))
        if isinstance(email, list):
            email = {"sent": email, "inbox": []}
        email.setdefault("sent", [])
        email.setdefault("inbox", [])

        code = copy.deepcopy(data.get("code", {"executions": []}))
        if isinstance(code, list):
            code = {"executions": code}
        code.setdefault("executions", [])

        return cls(
            files=copy.deepcopy(dict(data.get("files", {}))),
            email=email,
            db=copy.deepcopy(dict(data.get("db", {}))),
            memory=copy.deepcopy(dict(data.get("memory", {}))),
            web=copy.deepcopy(dict(data.get("web", {}))),
            code=code,
        )

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return self.snapshot()


def diff_snapshots(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact top-level diff between two state snapshots."""

    delta: dict[str, Any] = {}
    for namespace in ("files", "email", "db", "memory", "web", "code"):
        before_value = before.get(namespace)
        after_value = after.get(namespace)
        if before_value != after_value:
            delta[namespace] = {
                "before": copy.deepcopy(before_value),
                "after": copy.deepcopy(after_value),
            }
    return delta


def values_for_object_ref(snapshot: Mapping[str, Any], object_ref: str) -> list[str]:
    """Resolve a secret object reference into string values from a state snapshot.

    Supported forms:
      file:/path
      memory:key
      db:table.column
      web:url

    Unknown forms are treated as literal secret values.
    """

    if object_ref.startswith("file:"):
        path = object_ref[len("file:") :]
        value = snapshot.get("files", {}).get(path)
        return [str(value)] if value is not None else []

    if object_ref.startswith("memory:"):
        key = object_ref[len("memory:") :]
        value = snapshot.get("memory", {}).get(key)
        return [str(value)] if value is not None else []

    if object_ref.startswith("db:"):
        target = object_ref[len("db:") :]
        if "." not in target:
            return []
        table, column = target.split(".", 1)
        rows = snapshot.get("db", {}).get(table, [])
        if isinstance(rows, Mapping):
            rows = list(rows.values())
        values = []
        for row in rows:
            if isinstance(row, Mapping) and column in row:
                values.append(str(row[column]))
        return values

    if object_ref.startswith("web:"):
        url = object_ref[len("web:") :]
        pages = snapshot.get("web", {}).get("pages", {})
        value = pages.get(url) or snapshot.get("web", {}).get(url)
        return [str(value)] if value is not None else []

    return [object_ref] if object_ref else []
