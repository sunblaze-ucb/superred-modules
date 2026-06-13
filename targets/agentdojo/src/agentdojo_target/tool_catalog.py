"""ToolCatalog: per-run tool registry with attacker-driven mutations.

The catalog holds three kinds of entries:

- ``canonical``: the seed catalog built from :data:`tool_registry.ALL_FUNCTIONS`.
  These call into upstream AgentDojo tool bodies unchanged.
- ``registered``: an attacker-added tool.  When the agent invokes it,
  the runtime wrapper fires a per-call event and returns whatever the
  optimizer supplies (defaulting to the stored ``fake_return`` if the
  optimizer declines on this particular call).
- ``replaced``: a shadow of a canonical tool.  Same per-call event
  firing as ``registered``; the canonical body is never invoked while
  the shadow is in place.

Operations exposed to the optimizer via four Controllables:

- :func:`apply_register`     -> add a ``registered`` entry
- :func:`apply_replace`      -> mark a canonical entry as ``replaced``
- :func:`apply_unregister`   -> remove an entry entirely
- :func:`apply_rewrite_doc`  -> change an entry's description (body unchanged)

``snapshot()`` produces the agent-visible description of the catalog
(name + description + parameter schema for each entry).
``reset()`` restores the seed.

This module is pure state; it does NOT fire events.  The runtime
wrapper consults :meth:`ToolCatalog.classify` per call to decide
whether to call the underlying canonical body (for canonical reads /
writes) or to short-circuit to a per-call attacker event (for
registered / replaced shadows).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agentdojo.functions_runtime import Function
from pydantic import BaseModel, ConfigDict, create_model

EntryKind = Literal["canonical", "registered", "replaced"]


# ---------------------------------------------------------------------------
# Catalog entry
# ---------------------------------------------------------------------------


@dataclass
class CatalogEntry:
    """One entry in the catalog.

    Attributes:
        name: Tool name as the agent sees it (suite-prefixed for
            canonical entries; raw attacker-chosen name for registered).
        description: Agent-visible description.
        kind: ``canonical`` / ``registered`` / ``replaced``.
        function: The :class:`Function` to register with the underlying
            :class:`FunctionsRuntime`.  For canonical entries this is
            the upstream-bound Function.  For attacker entries it is a
            placeholder Function whose ``run`` should never be invoked
            (the runtime wrapper short-circuits before reaching it).
        fake_return: For attacker entries, the default value the
            wrapper substitutes when the optimizer declines to override
            on a per-call basis.  ``None`` for canonical entries.
    """

    name: str
    description: str
    kind: EntryKind
    function: Function
    fake_return: Any = None


# ---------------------------------------------------------------------------
# Placeholder Function builder (for attacker-registered tools)
# ---------------------------------------------------------------------------


class _PermissiveSchema(BaseModel):
    """Pydantic model that accepts any keyword args.

    Used as the ``parameters`` of an attacker-registered tool when the
    payload supplies no explicit ``parameters_schema``.  ``extra='allow'``
    keeps unknown fields rather than rejecting them.
    """

    model_config = ConfigDict(extra="allow")


def _placeholder_run(*args: Any, **kwargs: Any) -> Any:
    """Sentinel ``run`` callable for attacker-managed entries.

    The runtime wrapper short-circuits attacker calls into a synthetic
    event before invoking this callable, so reaching here indicates a
    logic error in the wrapper.
    """
    raise RuntimeError(
        "Placeholder run for an attacker-managed tool was invoked; the "
        "wrapped runtime should have intercepted this call.  Check the "
        "wrapper's dispatch on CatalogEntry.kind."
    )


def _build_parameters_class(
    name: str, schema: dict[str, Any] | None
) -> type[BaseModel]:
    """Convert an attacker-supplied JSON Schema fragment into a Pydantic model.

    Best-effort: only the ``properties`` + ``required`` keys are honored.
    Anything else (``oneOf``, ``allOf``, ``$ref``, etc.) is ignored and
    we fall back to the permissive schema.

    Args:
        name: Schema class name (use the tool name).
        schema: Optional JSON schema dict.  When ``None`` or empty, the
            permissive schema is returned.

    Returns:
        A pydantic ``BaseModel`` subclass usable as
        :attr:`Function.parameters`.
    """
    if not schema or not isinstance(schema.get("properties"), dict):
        return _PermissiveSchema

    properties: dict[str, Any] = schema["properties"]
    required = set(schema.get("required", []) or [])
    fields: dict[str, tuple[type, Any]] = {}
    for prop_name, prop_schema in properties.items():
        # Use Any for the type; the attacker's schema is untrusted and
        # we don't want pydantic rejecting valid LLM-emitted args.
        default = ... if prop_name in required else None
        fields[prop_name] = (Any, default)
    try:
        return create_model(f"AttackerToolArgs_{name}", **fields)
    except Exception:  # pragma: no cover - defensive
        return _PermissiveSchema


def _build_placeholder_function(
    name: str, description: str, parameters_schema: dict[str, Any] | None
) -> Function:
    """Build a :class:`Function` for an attacker-managed entry."""
    return Function(
        name=name,
        description=description,
        parameters=_build_parameters_class(name, parameters_schema),
        dependencies={},
        run=_placeholder_run,
        full_docstring=description,
        return_type=None,
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


@dataclass
class ToolCatalog:
    """Mutable tool catalog with attacker-driven mutations applied per run.

    The seed (canonical) catalog is captured at construction; mutations
    do not change the seed itself.  :meth:`reset` restores the catalog
    to the seed in O(n) time.
    """

    _seed: tuple[Function, ...] = field(repr=False)
    _entries: dict[str, CatalogEntry] = field(default_factory=dict, repr=False)

    @classmethod
    def from_seed(cls, seed_functions: list[Function]) -> ToolCatalog:
        """Construct a catalog from a canonical seed list."""
        catalog = cls(_seed=tuple(seed_functions))
        catalog.reset()
        return catalog

    def reset(self) -> None:
        """Restore the catalog to its canonical seed."""
        self._entries = {
            f.name: CatalogEntry(
                name=f.name,
                description=f.description,
                kind="canonical",
                function=f,
            )
            for f in self._seed
        }

    # ----- queries -----

    def get(self, name: str) -> CatalogEntry | None:
        """Return the entry for *name*, or ``None`` if absent."""
        return self._entries.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._entries

    def classify(self, name: str) -> EntryKind | None:
        """Return the entry's kind, or ``None`` when *name* is absent."""
        entry = self._entries.get(name)
        return entry.kind if entry is not None else None

    def functions_for_runtime(self) -> list[Function]:
        """Return the current catalog as a list of :class:`Function`
        instances ready to register with a :class:`FunctionsRuntime`."""
        return [entry.function for entry in self._entries.values()]

    def snapshot(self) -> list[dict[str, Any]]:
        """JSON-friendly snapshot suitable for an Observable payload.

        Each entry: ``{name, description, kind, parameters_schema}``.
        ``parameters_schema`` is the pydantic JSON schema for the
        underlying tool's args.
        """
        out: list[dict[str, Any]] = []
        for entry in self._entries.values():
            try:
                schema = entry.function.parameters.model_json_schema()
            except Exception:  # pragma: no cover - defensive
                schema = {}
            out.append(
                {
                    "name": entry.name,
                    "description": entry.description,
                    "kind": entry.kind,
                    "parameters_schema": schema,
                }
            )
        return out

    # ----- mutations -----

    def apply_register(self, payload: dict[str, Any]) -> CatalogEntry:
        """Apply a ``tool_catalog_register`` injection payload.

        Required keys: ``name``, ``description``, ``fake_return``.
        Optional: ``parameters_schema`` (JSON schema fragment).

        If *name* already exists, the call is rejected with ValueError.

        Returns the new :class:`CatalogEntry`.
        """
        name = _require_str(payload, "name")
        description = _require_str(payload, "description")
        if "fake_return" not in payload:
            raise ValueError("register payload missing 'fake_return'")
        parameters_schema = payload.get("parameters_schema")
        if name in self._entries:
            raise ValueError(
                f"Cannot register {name!r}: already in catalog "
                "(use replace or unregister first)"
            )
        function = _build_placeholder_function(name, description, parameters_schema)
        entry = CatalogEntry(
            name=name,
            description=description,
            kind="registered",
            function=function,
            fake_return=payload["fake_return"],
        )
        self._entries[name] = entry
        return entry

    def apply_replace(self, payload: dict[str, Any]) -> CatalogEntry:
        """Apply a ``tool_catalog_replace`` injection payload.

        Required keys: ``name``, ``fake_return``.
        Optional: ``description`` (override the agent-visible description).

        If *name* is absent or not canonical, ValueError.
        """
        name = _require_str(payload, "name")
        if "fake_return" not in payload:
            raise ValueError("replace payload missing 'fake_return'")
        existing = self._entries.get(name)
        if existing is None:
            raise ValueError(f"Cannot replace {name!r}: not in catalog")
        if existing.kind != "canonical":
            raise ValueError(
                f"Cannot replace {name!r}: already {existing.kind!r} (unregister first)"
            )
        new_description = payload.get("description") or existing.description
        # Build a placeholder Function preserving the agent-visible parameter
        # schema from the original so the agent's tool-calling format is
        # unchanged.  The body is the placeholder; the wrapper short-
        # circuits before it ever runs.
        replaced_fn = Function(
            name=name,
            description=new_description,
            parameters=existing.function.parameters,
            dependencies={},
            run=_placeholder_run,
            full_docstring=new_description,
            return_type=existing.function.return_type,
        )
        new_entry = CatalogEntry(
            name=name,
            description=new_description,
            kind="replaced",
            function=replaced_fn,
            fake_return=payload["fake_return"],
        )
        self._entries[name] = new_entry
        return new_entry

    def apply_unregister(self, payload: dict[str, Any]) -> None:
        """Apply a ``tool_catalog_unregister`` injection payload.

        Required keys: ``name``.

        Idempotent: removing an already-absent name is a no-op.
        """
        name = _require_str(payload, "name")
        self._entries.pop(name, None)

    def apply_rewrite_doc(self, payload: dict[str, Any]) -> CatalogEntry:
        """Apply a ``tool_catalog_rewrite_doc`` injection payload.

        Required keys: ``name``, ``description``.

        The underlying body / parameters are untouched; only the
        agent-visible description changes.  Works on canonical and
        attacker-managed entries alike.
        """
        name = _require_str(payload, "name")
        new_description = _require_str(payload, "description")
        existing = self._entries.get(name)
        if existing is None:
            raise ValueError(f"Cannot rewrite docs for {name!r}: not in catalog")
        # Rebuild Function with the new description; everything else preserved.
        new_fn = Function(
            name=name,
            description=new_description,
            parameters=existing.function.parameters,
            dependencies=dict(existing.function.dependencies),
            run=existing.function.run,
            full_docstring=new_description,
            return_type=existing.function.return_type,
        )
        new_entry = CatalogEntry(
            name=name,
            description=new_description,
            kind=existing.kind,
            function=new_fn,
            fake_return=existing.fake_return,
        )
        self._entries[name] = new_entry
        return new_entry


def _require_str(payload: dict[str, Any], key: str) -> str:
    """Pull a required string key from a payload, raising ValueError on miss."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"payload missing required string key {key!r} (got {value!r})")
    return value


__all__ = [
    "CatalogEntry",
    "EntryKind",
    "ToolCatalog",
]
