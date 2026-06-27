"""Local HTTP callback server for the superred injection plugin.

The OpenClaw plugin posts tool-call details here; the server returns
injection payloads that the plugin applies to tool parameters or results.

This is the Python-side counterpart to ``plugin/index.js``.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Union

from aiohttp import web

logger = logging.getLogger(__name__)

InjectionHandler = Callable[
    [str, str, dict[str, Any], str, Any],
    Union[dict[str, Any], None, Awaitable[Union[dict[str, Any], None]]],
]
"""(hook_type, tool_name, params, tool_call_id, result) -> payload or None.

Handler may be sync or async. Async handlers are awaited, so the
plugin's blocking ``before_tool_call`` POST receives the response only
after the optimizer-in-the-loop decision has been made.
"""


class InjectionServer:
    """Lightweight aiohttp server that receives plugin hook callbacks.

    The server runs on ``host:port`` and exposes ``POST /hook``.  For
    each request it calls the registered ``handler`` to decide whether
    to inject modified content.

    Args:
        handler: Callback invoked for each hook.  Return ``None`` to
            pass through, or a dict with injection instructions (see
            ``plugin/index.js`` for the expected shape).
        host: Bind address. ``127.0.0.1`` for a local gateway; ``0.0.0.0``
            when the gateway runs in Docker and reaches the host via
            ``host.docker.internal`` (the callback URL must advertise that
            alias, but the socket has to listen on a host-routable interface).
        port: Bind port. ``0`` (default) picks a free ephemeral port so
            multiple concurrent instances do not collide; read the chosen
            port back from :attr:`actual_port` after :meth:`start`.
    """

    def __init__(
        self,
        handler: InjectionHandler,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._handler = handler
        self._host = host
        self._port = port
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    @property
    def actual_port(self) -> int:
        """The bound port (resolved after :meth:`start` when ``port=0``)."""
        return self._port

    async def start(self) -> None:
        """Start the server in the background."""
        app = web.Application()
        app.router.add_post("/hook", self._handle_hook)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        # Resolve the actual port when 0 (ephemeral) was requested.
        server = self._site._server  # type: ignore[attr-defined]
        sockets = getattr(server, "sockets", None)
        if sockets:
            self._port = sockets[0].getsockname()[1]

        logger.info(
            "Injection server listening on http://%s:%d", self._host, self._port,
        )

    async def stop(self) -> None:
        """Shut down the server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None

    async def _handle_hook(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": "invalid json"}, status=400,
            )

        hook_type = body.get("hook", "")
        # The real plugin sends `toolName`; accept `tool` as a fallback.
        tool_name = body.get("toolName") or body.get("tool", "")
        params = body.get("params", {})
        tool_call_id = body.get("toolCallId", "") or ""
        result = body.get("result")

        try:
            injection = self._handler(
                hook_type, tool_name, params, tool_call_id, result,
            )
            if inspect.isawaitable(injection):
                injection = await injection
        except Exception:
            logger.exception("Injection handler error")
            injection = None

        if injection is None:
            return web.json_response({})

        return web.json_response(injection)

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"
