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

import json
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

    def _upstream_chat_completions_url(self) -> str:
        """Build the upstream ``/chat/completions`` URL.

        Mirrors ``config.py``'s ``build_gateway_config`` normalization
        (append ``/v1`` only if the caller's base URL doesn't already end in
        it) so a ``provider_base_url`` that already contains a trailing
        ``/v1`` segment - as OpenAI-compatible endpoints from providers other
        than OpenAI itself commonly do, e.g. Gemini's
        ``https://generativelanguage.googleapis.com/v1beta/openai/v1`` -
        doesn't get double-appended into ``.../v1/v1/chat/completions``
        (verified live: this 404s).
        """
        base = self._upstream_base_url
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    @staticmethod
    def _normalize_upstream_model(model: str) -> str:
        """Strip OpenClaw's provider-qualified id before upstream relay.

        OpenClaw sends ``google/gemini-2.5-flash``; Gemini's OpenAI-compatible
        endpoint expects ``gemini-2.5-flash`` (verified live: the qualified
        form 400s and returns a JSON *array* error body).
        """
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    @staticmethod
    def _coerce_completion_response(payload: Any, *, status: int) -> dict[str, Any]:
        """Normalize upstream success/error payloads to a response dict."""
        if isinstance(payload, dict):
            if "error" in payload or "choices" in payload:
                return payload
            return {
                "error": {
                    "message": json.dumps(payload)[:4000],
                    "code": status,
                },
            }
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict) and "error" in first:
                error = first["error"]
                if isinstance(error, dict):
                    return {"error": error}
                return {"error": {"message": str(error), "code": status}}
        return {
            "error": {
                "message": str(payload)[:4000] or f"HTTP {status}",
                "code": status,
            },
        }

    async def _read_upstream_json(
        self, resp: aiohttp.ClientResponse,
    ) -> dict[str, Any]:
        """Parse the upstream body as JSON, tolerating a non-JSON error body
        (some providers return plain-text/HTML for 4xx/5xx responses, which
        would otherwise raise and crash the handler with an unrelated
        ``ContentTypeError``)."""
        try:
            payload = await resp.json(content_type=None)
        except (aiohttp.ContentTypeError, ValueError):
            text = await resp.text()
            return {"error": {"message": text[:4000] or f"HTTP {resp.status}", "code": resp.status}}
        return self._coerce_completion_response(payload, status=resp.status)

    async def _consume_sse(
        self,
        resp: aiohttp.ClientResponse,
        *,
        forward_to: aiohttp.web.StreamResponse | None,
        model: str,
    ) -> tuple[dict[str, Any], bytes | None]:
        """Parse an upstream SSE chat-completion stream into a single
        non-streaming-shaped response dict (for injection/recording).

        If ``forward_to`` is given, every genuine content/tool-call delta is
        relayed to it live, as it arrives - true incremental streaming, not
        buffered-then-replayed. The terminal (``finish_reason``) frame is
        *not* written here; it is returned as ``pending_finish_frame`` so the
        caller can splice in a response-injection delta chunk, if any,
        strictly *before* it (matching where ``_inject_response`` appends
        injected text: after all real content). This keeps genuine model
        output fully live-streamed while still injecting correctly.
        """
        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        meta: dict[str, Any] = {}
        usage: dict[str, Any] | None = None
        pending_finish_frame: bytes | None = None

        def meta_base() -> dict[str, Any]:
            return {
                "id": meta.get("id", "chatcmpl-proxy"),
                "object": "chat.completion.chunk",
                "created": meta.get("created") or int(time.time()),
                "model": meta.get("model", model),
            }

        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            if "error" in chunk:
                if forward_to is not None:
                    await forward_to.write(f"data: {json.dumps(chunk)}\n\n".encode())
                return {"error": chunk["error"]}, None

            for key in ("id", "object", "created", "model"):
                if chunk.get(key) is not None:
                    meta[key] = chunk[key]
            if chunk.get("usage") is not None:
                usage = chunk["usage"]

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")

            content_piece = delta.get("content")
            if content_piece:
                content_parts.append(content_piece)
            tc_pieces = delta.get("tool_calls") or []
            for tc in tc_pieces:
                idx = tc.get("index", 0)
                slot = tool_calls.setdefault(
                    idx,
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("type"):
                    slot["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]

            if forward_to is not None and (content_piece or tc_pieces or delta.get("role")):
                live_chunk = {
                    **meta_base(),
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                await forward_to.write(f"data: {json.dumps(live_chunk)}\n\n".encode())

            if finish_reason:
                final_chunk = {
                    **meta_base(),
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                pending_finish_frame = f"data: {json.dumps(final_chunk)}\n\n".encode()

        final_content = "".join(content_parts) or None
        message: dict[str, Any] = {"role": "assistant", "content": final_content}
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

        response_raw: dict[str, Any] = {
            "id": meta.get("id", "chatcmpl-proxy"),
            "object": "chat.completion",
            "created": meta.get("created"),
            "model": meta.get("model", model),
            "choices": [{"index": 0, "message": message, "finish_reason": None}],
            "usage": usage or {},
        }
        return response_raw, pending_finish_frame

    async def _handle_completions(
        self, request: aiohttp.web.Request,
    ) -> aiohttp.web.StreamResponse:
        """Intercept chat completions: record, optionally modify, forward.

        OpenClaw's real openai-completions client always sends
        `stream: true` (verified against openclaw/openclaw
        src/llm/providers/openai-completions.ts), and real providers honor
        it with a genuine SSE reply (`Content-Type: text/event-stream`) -
        unlike our stub test LLM servers, which ignore the flag and always
        reply with plain JSON. The upstream response shape is auto-detected
        (not assumed from the request), so both cases work: a real
        provider's SSE stream is relayed to the client live, chunk by
        chunk, as it arrives (true incremental delivery is preserved end to
        end - forcing non-streaming upstream would silently degrade
        ``assistant_stream`` from live token-by-token deltas to a single
        chunk delivered only once generation fully finishes, verified live
        against Gemini: 3 incremental deltas direct vs. 1 batched delta
        through an earlier version of this proxy that forced
        `stream: false` upstream); a stub's plain JSON reply is still
        forwarded as plain JSON (OpenClaw's client already tolerates that
        for a `stream: true` request, proven by every stub-LLM test in this
        suite), so nothing here depends on real API keys to test.
        """
        if not self._authorized(request):
            return aiohttp.web.json_response({"error": "unauthorized"}, status=401)

        body = await request.json()
        messages = body.get("messages", [])
        model = body.get("model", "unknown")
        client_wants_stream = bool(body.get("stream"))

        if self.system_prompt_injection is not None:
            self._inject_system_prompt(messages)

        upstream_body = dict(body)
        upstream_body["model"] = self._normalize_upstream_model(str(model))

        upstream_url = self._upstream_chat_completions_url()
        headers = {
            "Authorization": f"Bearer {self._upstream_api_key}",
            "Content-Type": "application/json",
        }

        assert self._session is not None
        resp = await self._session.post(
            upstream_url, json=upstream_body, headers=headers,
        )
        stream_resp: aiohttp.web.StreamResponse | None = None
        pending_finish_frame: bytes | None = None
        try:
            upstream_status = resp.status
            is_upstream_sse = "text/event-stream" in (resp.content_type or "")
            if is_upstream_sse:
                if client_wants_stream:
                    stream_resp = aiohttp.web.StreamResponse(
                        status=upstream_status,
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                        },
                    )
                    await stream_resp.prepare(request)
                resp_body, pending_finish_frame = await self._consume_sse(
                    resp, forward_to=stream_resp, model=model,
                )
            else:
                resp_body = await self._read_upstream_json(resp)
        finally:
            resp.release()

        resp_body = self._coerce_completion_response(resp_body, status=upstream_status)

        # Rewrite the model's reply before it reaches the agent. Done before
        # recording so the trace reflects what the agent actually saw. For
        # the live-relay case, the wire already carries the real content;
        # the injection delta is appended to the wire separately below
        # (after the finish frame is held back), so this mutation only
        # affects the *recorded* text, not a second copy sent over HTTP.
        if self.response_injection is not None and "error" not in resp_body:
            self._inject_response(resp_body)

        response_text = ""
        choices = resp_body.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            # `or ""`: a tool-calls-only assistant message has content=None.
            response_text = msg.get("content") or ""

        usage = resp_body.get("usage", {})
        self.records.append(ModelCallRecord(
            timestamp_ms=int(time.time() * 1000),
            request_messages=messages,
            request_model=model,
            response_text=response_text,
            response_raw=resp_body,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        ))

        if stream_resp is not None:
            if "error" in resp_body:
                await stream_resp.write(
                    f"data: {json.dumps(resp_body)}\n\n".encode(),
                )
                await stream_resp.write_eof()
                return stream_resp
            if self.response_injection is not None:
                inject_chunk = {
                    "id": resp_body.get("id", "chatcmpl-proxy"),
                    "object": "chat.completion.chunk",
                    "created": resp_body.get("created") or int(time.time()),
                    "model": resp_body.get("model", model),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "\n" + self.response_injection},
                            "finish_reason": None,
                        },
                    ],
                }
                await stream_resp.write(f"data: {json.dumps(inject_chunk)}\n\n".encode())
            if pending_finish_frame is not None:
                await stream_resp.write(pending_finish_frame)
            await stream_resp.write(b"data: [DONE]\n\n")
            await stream_resp.write_eof()
            return stream_resp

        return aiohttp.web.json_response(resp_body, status=upstream_status)

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
        assert self.response_injection is not None
        for choice in resp_body.get("choices", []):
            msg = choice.get("message")
            if isinstance(msg, dict):
                msg["content"] = (msg.get("content") or "") + "\n" + self.response_injection
