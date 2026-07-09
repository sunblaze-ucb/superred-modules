"""Collaborator interfaces the agent-agnostic base depends on (dependency-injected).

The base wires the superred lifecycle around three collaborators typed here:
an :class:`EnvStack` (the Docker env + injection containers), an
:class:`MCPProxy` (the host-side seam that observes/controls env tool calls), and
an :class:`EnvInjector` (the DTAP environment-write vector). Concrete
implementations live in ``dtap_scaffold.docker.lifecycle`` / ``mcp_proxy`` /
``injection``; tests substitute fakes. Typing only (``Protocol``), so any
duck-typed object satisfies them.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse

from dtap_scaffold.types import EnvHandle, InjectionPoint, ProxyTool

EmitFn = Callable[[Event], None]
SendEventFn = Callable[[Event], Awaitable[EventResponse]]


@runtime_checkable
class EnvStack(Protocol):
    """The per-instance Docker environment (env + injection containers)."""

    async def up(self) -> EnvHandle: ...
    async def reset(self) -> None: ...
    async def down(self) -> None: ...


@runtime_checkable
class EnvInjector(Protocol):
    """The DTAP environment-write vector via the ``<server>-injection`` MCP tools."""

    async def snapshot(self, point: InjectionPoint) -> str:
        """Genuine current backend content for the PostCall event's ``answer``."""
        ...

    async def apply(self, point: InjectionPoint, value: str) -> None:
        """Write attacker *value* into the live backend via the injection tool."""
        ...


@runtime_checkable
class MCPProxy(Protocol):
    """Host-side proxy fronting the env MCP servers: the observe/control seam.

    The agent (in its container) connects to :meth:`start`'s URL and issues every
    env tool call through the proxy. Each call is observed (one ObservableEvent)
    and offered for return-tampering via the ``env_tool`` PostCall Controllable
    scoped to the tool's authorization node (see :mod:`dtap_scaffold.tool_trees`).
    Tool DESCRIPTIONS are edited in :meth:`list_tools` from the PreCall tool-vector
    injections.
    """

    async def start(self, server_urls: dict[str, str]) -> str:
        """Start the proxy fronting *server_urls*; return the agent-facing URL."""
        ...

    def bind(self, emit: EmitFn, send_event: SendEventFn) -> None:
        """Bind this run's event callbacks (called at the start of each run)."""
        ...

    def set_tool_description_edits(self, edits: list[dict[str, Any]]) -> None:
        """Set PreCall tool-vector edits: ``[{server, tool, mode, content}]``."""
        ...

    def set_env_tool_controllables(
        self,
        by_server_tool: dict[str, dict[str, Controllable]],
        defaults: dict[str, Controllable],
    ) -> None:
        """Set the ``env_tool`` Controllables used for PostCall firing.

        *by_server_tool*: ``server -> {tool -> node Controllable}``; *defaults*:
        ``server -> the tools.<server> root Controllable`` (fallback for tools
        absent from the tree).
        """
        ...

    def list_tools(self, server: str) -> list[ProxyTool]:
        """Tools for *server*, with description edits applied (for the listing)."""
        ...

    def tool_catalogue(self) -> dict[str, list[dict[str, Any]]]:
        """The genuine per-tool catalogue after :meth:`start`:
        ``{server: [{"name", "description", "inputSchema"}]}`` (pre-edit), the
        observability surface the optimizer reads to understand the tool space."""
        ...

    async def handle_tool_call(
        self, server: str, tool: str, params: dict[str, Any]
    ) -> tuple[str, bool]:
        """The chokepoint: observe, fire env_tool PostCall, forward, return ``(text, is_error)``.

        Called by the HTTP layer for the real agent; called directly by tests.
        """
        ...

    async def stop(self) -> None: ...


__all__ = [
    "EmitFn",
    "SendEventFn",
    "EnvStack",
    "EnvInjector",
    "MCPProxy",
]
