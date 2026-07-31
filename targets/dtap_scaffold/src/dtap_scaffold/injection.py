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
        for tool, kwargs in _parse_injection_calls(value):
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


def _parse_injection_calls(value: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an ``env_inject`` value into ``[(tool, kwargs), ...]`` calls.

    The env_inject controllable's ``value_type`` is JSON:
    ``{"injection_mcp_tool": "<server>:<tool>", "kwargs": {...}}`` -> the tool name is
    the part after the colon. A JSON list of such objects produces several calls. A
    malformed value (non-JSON, a non-dict, or a dict without ``injection_mcp_tool``)
    yields no call, and a non-dict ``kwargs`` is coerced to an empty mapping, so a
    wrong-shaped value never aborts the run -- matching upstream
    ``get_env_injections_from_attack`` (the tool identity always rides in the value).
    """
    try:
        parsed: Any = json.loads(value)
    except (ValueError, TypeError, RecursionError):
        # json.loads raises JSONDecodeError (a ValueError) on bad JSON, a bare ValueError
        # on an oversized int literal (>4300 digits), TypeError on non-str input, and
        # RecursionError on deeply-nested JSON -- catch them all so no value aborts the run.
        return []

    specs = parsed if isinstance(parsed, list) else [parsed]
    calls: list[tuple[str, dict[str, Any]]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        injection_mcp_tool = spec.get("injection_mcp_tool")
        if injection_mcp_tool:
            tool = str(injection_mcp_tool).split(":", 1)[-1]
            raw_kwargs = spec.get("kwargs")
            kwargs = raw_kwargs if isinstance(raw_kwargs, dict) else {}
            calls.append((tool, kwargs))
    return calls


def build_env_injection(value: str, template: dict[str, Any] | None) -> str | None:
    """Resolve an ``env_inject`` injected *value* into a structured injection payload.

    Two paths, so a generic attacker can drive this vector with plain text while the
    faithful structured form still works:

    - a value that already parses to one or more ``{injection_mcp_tool, kwargs}`` calls
      is returned unchanged (the upstream-faithful structured path);
    - otherwise, if the task supplied a per-server *template* (its reference inject
      tool + routing ``kwargs`` + the ``content_field`` name), the plain-text *value* is
      spliced into that content field, carrying every routing kwarg verbatim -- so only
      the attacker-controlled CONTENT varies, exactly the axis DTAP's red-teamer varies.

    Returns the JSON payload to write, or ``None`` when nothing should be written (no
    template, or an empty payload) -- preserving the clean baseline and the honest
    decline for servers whose inject tool has no placeable content field.
    """
    if _parse_injection_calls(value):
        return value
    if not template:
        return None
    field = template.get("content_field")
    content = value if isinstance(value, str) else ""
    if not field or not content.strip():
        return None
    kwargs = {**(template.get("kwargs") or {}), field: content}
    return json.dumps(
        {"injection_mcp_tool": template.get("injection_mcp_tool"), "kwargs": kwargs}
    )


__all__ = ["McpEnvInjector", "build_env_injection"]
