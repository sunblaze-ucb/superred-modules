"""The DTAP environment-write vector: the concrete :class:`~dtap_scaffold.protocols.EnvInjector`.

DTAP's *environment* injection vector writes attacker-controlled data straight
into a live backend (an email into the mailbox, a comment onto a record, a file
onto the shared volume) via dedicated ``<server>-injection`` MCP servers whose
only tools are write-only ``inject_*`` calls (see
``dt_arena/injection_mcp_server/<domain>/env_injection.py`` upstream). The agent
then encounters that data later through its ordinary env tools -- an indirect
prompt injection that lands in the world, not just in a single tool return.

:class:`McpEnvInjector` is what the agent-agnostic base calls for the per-server
``env_inject`` PostCall controllable:

- :meth:`snapshot` supplies the PostCall event's genuine ``answer``. The injection
  servers expose no read tool, so there is no genuine backend content to read
  through this vector; it returns ``""`` (best-effort). The agent observes any
  injected data later through its normal env tools (the proxy's ``env_tool`` path).
- :meth:`apply` takes the optimizer's injected value and executes the actual
  ``inject_*`` MCP call against ``injection_server_urls[point.server]``. The value
  is the structured ``{"injection_mcp_tool": "<server>:<tool>", "kwargs": {...}}``
  form (a list of these is also accepted for several writes), matching upstream
  ``get_env_injections_from_attack`` / ``apply_environment_injections_async``.

The fastmcp client call is isolated in :meth:`_call`, the seam offline tests
monkeypatch, so no network or ``fastmcp`` import happens under test.
"""

from __future__ import annotations

import json
from typing import Any

from dtap_scaffold.types import InjectionPoint


class McpEnvInjector:
    """Execute DTAP environment writes through the ``<server>-injection`` MCP tools."""

    def __init__(self, injection_server_urls: dict[str, str]) -> None:
        self._urls: dict[str, str] = dict(injection_server_urls)

    async def snapshot(self, point: InjectionPoint) -> str:
        """Genuine current backend content for the PostCall ``answer``.

        The DTAP injection servers are write-only (only ``inject_*`` tools), so
        there is no read path through this vector; returns ``""``. The injected
        data surfaces later via the agent's normal env tools.
        """
        return ""

    async def apply(self, point: InjectionPoint, value: str) -> None:
        """Write the attacker *value* into the live backend for *point.server*.

        *value* is parsed into one or more ``(tool, kwargs)`` injection calls and
        each is dispatched to ``injection_server_urls[point.server]``. Writing is
        best-effort: an unknown server or a failing call is swallowed (the attack
        simply does not land) rather than aborting the run, matching upstream's
        per-injection error handling.
        """
        url = self._urls.get(point.server)
        if not url:
            return
        for tool, kwargs in _parse_injection_calls(value, point):
            if not tool:
                continue
            try:
                await self._call(url, tool, kwargs)
            except Exception:  # noqa: BLE001 - best-effort write, like upstream
                continue

    async def _call(self, url: str, tool: str, kwargs: dict[str, Any]) -> Any:
        """Call one ``inject_*`` tool on the injection server (the monkeypatch seam).

        Production path uses a fastmcp client, exactly as upstream
        ``apply_environment_injections_async``. Offline tests replace this method.
        """
        from fastmcp import Client  # lazy: container/runtime-only dependency

        async with Client(url, timeout=30.0) as client:
            return await client.call_tool(tool, kwargs)


def _parse_injection_calls(value: str, point: InjectionPoint) -> list[tuple[str, dict[str, Any]]]:
    """Parse an ``env_inject`` value into ``[(tool, kwargs), ...]`` calls.

    Primary form (the env_inject controllable's ``value_type`` is JSON):
    ``{"injection_mcp_tool": "<server>:<tool>", "kwargs": {...}}`` -> the tool name
    is the part after the colon (``injection_server_urls[point.server]`` supplies
    the URL). A JSON list of such objects produces several calls.

    Fallbacks (kept so a malformed value never aborts the run):
      - a JSON object WITHOUT ``injection_mcp_tool`` is treated as the ``kwargs``
        for ``point.point`` (when that names a concrete inject tool);
      - a bare non-JSON string is forwarded as ``{"content": value}`` to
        ``point.point`` (when concrete).
    The ``"all"`` sentinel for ``point.point`` means "the value carries the tool",
    so these fallbacks no-op for it -- only the structured form applies.
    """
    try:
        parsed: Any = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if parsed is None:
        tool = _concrete_point_tool(point)
        return [(tool, {"content": value})] if tool else []

    specs = parsed if isinstance(parsed, list) else [parsed]
    calls: list[tuple[str, dict[str, Any]]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        injection_mcp_tool = spec.get("injection_mcp_tool")
        if injection_mcp_tool:
            tool = str(injection_mcp_tool).split(":", 1)[-1]
            kwargs = spec.get("kwargs") or {}
            calls.append((tool, dict(kwargs)))
        else:
            tool = _concrete_point_tool(point)
            if tool:
                calls.append((tool, dict(spec)))
    return calls


def _concrete_point_tool(point: InjectionPoint) -> str | None:
    """The point's inject tool name, or None when it is the ``"all"`` sentinel/empty."""
    if point.point and point.point != "all":
        return point.point
    return None


__all__ = ["McpEnvInjector"]
