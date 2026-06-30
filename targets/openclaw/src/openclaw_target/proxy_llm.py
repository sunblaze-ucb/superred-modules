"""LLM proxy for intercepting model calls.

Sits between OpenClaw and the real LLM provider, forwarding
OpenAI-compatible ``/v1/chat/completions`` requests while:

- Recording every model request and response for trace generation
- Optionally modifying the system prompt (controllable)
- Optionally modifying the assistant response text (controllable)

Start this proxy and point OpenClaw's provider base URL at it.
The proxy forwards to the real provider transparently.

Requires ``aiohttp``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import aiohttp.web

logger = logging.getLogger(__name__)


@dataclass
class ModelCallRecord:
    """A recorded model request/response pair."""

    timestamp_ms: int
    request_messages: list[dict[str, Any]]
    request_model: str
    response_text: str
    response_raw: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProxy:
    """Local HTTP proxy for OpenAI-compatible chat completions.

    Args:
        upstream_base_url: The real provider URL to forward to
            (e.g. ``https://api.openai.com``).
        upstream_api_key: API key for the upstream provider.
        host: Bind address for the proxy server.
        port: Port for the proxy server (0 = auto-assign).
        inbound_token: If set, every inbound request must present
            ``Authorization: Bearer <inbound_token>``; otherwise it is
            rejected with 401. This is what stops the proxy from being an
            *unauthenticated* open relay when it has to bind a non-loopback
            interface (e.g. so a container can reach it via
            ``host.docker.internal``). The gateway sends this token because it
            is configured as the provider ``apiKey``; the proxy forwards
            upstream with the real ``upstream_api_key`` instead.
    """

    def __init__(
        self,
        upstream_base_url: str,
        upstream_api_key: str,
        host: str = "127.0.0.1",
        port: int = 0,
        inbound_token: str | None = None,
    ) -> None:
        self._upstream_base_url = upstream_base_url.rstrip("/")
        self._upstream_api_key = upstream_api_key
        self._host = host
        self._port = port
        self._inbound_token = inbound_token

        self._app: aiohttp.web.Application | None = None
        self._runner: aiohttp.web.AppRunner | None = None
        self._site: aiohttp.web.TCPSite | None = None
        self._session: aiohttp.ClientSession | None = None

        self.records: list[ModelCallRecord] = []
        self.system_prompt_injection: str | None = None
        self.response_injection: str | None = None

    @property
    def proxy_base_url(self) -> str:
        """The base URL other services should point at."""
        return f"http://{self._host}:{self._port}"

    @property
    def actual_port(self) -> int:
        return self._port

    async def start(self) -> int:
        """Start the proxy server. Returns the bound port."""
        self._session = aiohttp.ClientSession()
        self._app = aiohttp.web.Application()
        self._app.router.add_post(
            "/v1/chat/completions", self._handle_completions,
        )
        # Catch-all for other endpoints — pass through
        self._app.router.add_route("*", "/{path:.*}", self._handle_passthrough)

        self._runner = aiohttp.web.AppRunner(self._app)
        await self._runner.setup()
        self._site = aiohttp.web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        # Resolve actual port if 0 was requested
        for sock in self._site._server.sockets:  # type: ignore[union-attr]
            self._port = sock.getsockname()[1]
            break

        logger.info(
            "LLM proxy listening on %s, forwarding to %s",
            self.proxy_base_url, self._upstream_base_url,
        )
        return self._port

    async def stop(self) -> None:
        """Stop the proxy server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        if self._session:
            await self._session.close()
        self._site = None
        self._runner = None
        self._session = None

    def _authorized(self, request: aiohttp.web.Request) -> bool:
        """Whether the inbound request carries the required bearer token."""
        if self._inbound_token is None:
            return True
        header = request.headers.get("Authorization", "")
        expected = f"Bearer {self._inbound_token}"
        return header == expected

    async def _handle_completions(
        self, request: aiohttp.web.Request,
    ) -> aiohttp.web.Response:
        """Intercept chat completions: record, optionally modify, forward."""
        if not self._authorized(request):
            return aiohttp.web.json_response({"error": "unauthorized"}, status=401)

        body = await request.json()
        messages = body.get("messages", [])
        model = body.get("model", "unknown")

        if self.system_prompt_injection is not None:
            self._inject_system_prompt(messages)

        upstream_url = f"{self._upstream_base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._upstream_api_key}",
            "Content-Type": "application/json",
        }

        assert self._session is not None
        async with self._session.post(
            upstream_url, json=body, headers=headers,
        ) as resp:
            resp_body = await resp.json()

        # Rewrite the model's reply before it reaches the agent. Done before
        # recording so the trace reflects what the agent actually saw.
        if self.response_injection is not None:
            self._inject_response(resp_body)

        response_text = ""
        input_tokens = 0
        output_tokens = 0

        choices = resp_body.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            response_text = msg.get("content", "")

        usage = resp_body.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        self.records.append(ModelCallRecord(
            timestamp_ms=int(time.time() * 1000),
            request_messages=messages,
            request_model=model,
            response_text=response_text,
            response_raw=resp_body,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ))

        return aiohttp.web.json_response(resp_body, status=resp.status)

    async def _handle_passthrough(
        self, request: aiohttp.web.Request,
    ) -> aiohttp.web.Response:
        """Forward non-completions requests unchanged."""
        if not self._authorized(request):
            return aiohttp.web.json_response({"error": "unauthorized"}, status=401)
        path = request.match_info.get("path", "")
        upstream_url = f"{self._upstream_base_url}/{path}"
        headers = dict(request.headers)
        headers["Authorization"] = f"Bearer {self._upstream_api_key}"
        headers.pop("Host", None)

        body = await request.read()

        assert self._session is not None
        async with self._session.request(
            request.method, upstream_url,
            headers=headers, data=body,
        ) as resp:
            resp_body = await resp.read()
            return aiohttp.web.Response(
                body=resp_body,
                status=resp.status,
                content_type=resp.content_type,
            )

    def _inject_system_prompt(self, messages: list[dict[str, Any]]) -> None:
        """Append injection content to the system prompt message."""
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = msg.get("content", "") + "\n" + self.system_prompt_injection
                return
        messages.insert(0, {
            "role": "system",
            "content": self.system_prompt_injection,
        })

    def _inject_response(self, resp_body: dict[str, Any]) -> None:
        """Append injection content to the assistant message in the response."""
        for choice in resp_body.get("choices", []):
            msg = choice.get("message")
            if isinstance(msg, dict):
                msg["content"] = (msg.get("content") or "") + "\n" + self.response_injection
