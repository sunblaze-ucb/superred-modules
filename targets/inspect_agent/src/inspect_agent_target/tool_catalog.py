"""Mutable tool catalogue for the inspect-agent target.

Holds the current tool set as editable inspect ``ToolDef``s and applies attacker
catalogue edits fired through the tool-catalogue Controllables:

- register: add a new attacker-defined tool with a fixed canned return;
- replace: shadow an existing tool's behaviour with a canned return;
- unregister: remove a tool;
- rewrite_doc: change a tool's description without changing its behaviour.

Malformed edits are logged and ignored, never raised, so a bad attacker payload
cannot abort the agent run (the run's progress matters more than payload
correctness). The initial catalogue is seeded from the Task-configured tool
names via the target's resolver.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from inspect_ai.tool import Tool, ToolDef, ToolParams

logger = logging.getLogger(__name__)

ToolResolver = Callable[[str], Tool]


_JSON_TO_PY: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


def _canned_tool(return_value: Any, params: ToolParams) -> Callable[..., Any]:
    """Build an async tool callable that returns a fixed string.

    The callable's signature must match the declared schema parameters, because
    inspect binds a model's tool-call arguments against the callable's *actual*
    signature at execution time (and requires a type annotation per parameter).
    A generic ``**kwargs`` does not work: inspect treats the var-keyword
    parameter as a single required parameter ("Required parameter _kwargs not
    provided"). So we synthesise a function whose parameters are exactly the
    schema's property names (typed from the schema, each optional), and which
    ignores them and returns the canned value.
    """
    parts: list[str] = []
    for name, schema in params.properties.items():
        if not name.isidentifier():
            continue
        jtype = getattr(schema, "type", None)
        pytype = _JSON_TO_PY.get(jtype, "str") if isinstance(jtype, str) else "str"
        parts.append(f"{name}: {pytype} = None")
    src = f"async def _canned({', '.join(parts)}) -> str:\n    return _RETVAL\n"
    namespace: dict[str, Any] = {"_RETVAL": str(return_value)}
    exec(src, namespace)  # noqa: S102 - fixed-return stub with a schema-shaped signature
    return namespace["_canned"]  # type: ignore[no-any-return]


class ToolCatalog:
    """An ordered, mutable set of tools addressed by name."""

    def __init__(self) -> None:
        self._defs: dict[str, ToolDef] = {}

    @classmethod
    def seed(cls, resolver: ToolResolver, names: list[str]) -> ToolCatalog:
        """Build the initial catalogue by resolving *names* to tools."""
        catalog = cls()
        for name in names:
            catalog._defs[name] = ToolDef(resolver(name))
        return catalog

    def tools(self) -> list[Tool]:
        """The current tools, ready to pass to ``Model.generate``."""
        return [td.as_tool() for td in self._defs.values()]

    def names(self) -> list[str]:
        return list(self._defs.keys())

    def snapshot(self) -> list[dict[str, Any]]:
        """JSON-friendly listing of the catalogue (for the observable)."""
        return [
            {
                "name": td.name,
                "description": td.description,
                "parameters_schema": td.parameters.model_dump(),
            }
            for td in self._defs.values()
        ]

    # -- attacker edits (all log-and-ignore on malformed input) ---------------

    def apply_register(self, payload: dict[str, Any]) -> None:
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            logger.warning("tool_catalog_register: missing/invalid 'name'")
            return
        description = payload.get("description") or f"Tool {name}."
        schema = payload.get("parameters_schema") or {}
        try:
            params = ToolParams.model_validate(schema) if schema else ToolParams()
        except Exception as exc:
            logger.warning("tool_catalog_register %r: bad parameters_schema: %s", name, exc)
            return
        self._defs[name] = ToolDef(
            _canned_tool(payload.get("fake_return", ""), params),
            name=name,
            description=str(description),
            parameters=params,
        )

    def apply_replace(self, payload: dict[str, Any]) -> None:
        name = payload.get("name")
        if not isinstance(name, str) or name not in self._defs:
            logger.warning("tool_catalog_replace: unknown/invalid 'name' %r", name)
            return
        existing = self._defs[name]
        description = payload.get("description") or existing.description
        self._defs[name] = ToolDef(
            _canned_tool(payload.get("fake_return", ""), existing.parameters),
            name=name,
            description=str(description),
            parameters=existing.parameters,
        )

    def apply_unregister(self, payload: dict[str, Any]) -> None:
        name = payload.get("name")
        if isinstance(name, str) and name in self._defs:
            del self._defs[name]
        else:
            logger.warning("tool_catalog_unregister: unknown/invalid 'name' %r", name)

    def apply_rewrite_doc(self, payload: dict[str, Any]) -> None:
        name = payload.get("name")
        description = payload.get("description")
        if not isinstance(name, str) or name not in self._defs:
            logger.warning("tool_catalog_rewrite_doc: unknown/invalid 'name' %r", name)
            return
        if not isinstance(description, str):
            logger.warning("tool_catalog_rewrite_doc %r: missing 'description'", name)
            return
        self._defs[name].description = description


__all__ = ["ToolCatalog", "ToolResolver"]
