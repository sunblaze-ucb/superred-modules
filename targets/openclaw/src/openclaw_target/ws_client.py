"""Async WebSocket client for the OpenClaw Gateway protocol.

Handles the connect handshake, request/response dispatch, and event
streaming required to drive the Gateway from Python.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import websockets
import websockets.asyncio.client

logger = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    """A single event from an agent run stream."""

    stream: str
    payload: dict[str, Any]
    raw: dict[str, Any]


@dataclass
class AgentRunResult:
    """Aggregated result of a completed agent run."""

    run_id: str
    status: str
    assistant_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    events: list[AgentEvent] = field(default_factory=list)
    error: str | None = None


class OpenClawWSClient:
    """Async WebSocket client for the OpenClaw Gateway.

    Args:
        gateway_url: WebSocket URL (e.g. ``ws://127.0.0.1:18789``).
        auth_token: Shared-secret auth token for the Gateway.
        device_id: Stable device identifier for this client.
    """

    def __init__(
        self,
        gateway_url: str,
        auth_token: str,
        device_id: str | None = None,
    ) -> None:
        self._url = gateway_url
        self._auth_token = auth_token
        self._device_id = device_id or f"superred-{uuid.uuid4().hex[:12]}"
        self._ws: websockets.asyncio.client.ClientConnection | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._event_listeners: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._connected = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> dict[str, Any]:
        """Open the WebSocket and complete the Gateway handshake.

        Returns:
            The ``hello-ok`` payload from the Gateway.

        Raises:
            ConnectionError: If the handshake fails.
        """
        self._ws = await websockets.asyncio.client.connect(self._url)
        self._reader_task = asyncio.create_task(self._read_loop())

        challenge_event = await self._wait_for_event("connect.challenge", timeout=10)
        nonce = challenge_event.get("payload", {}).get("nonce", "")

        connect_id = self._next_id()
        connect_req: dict[str, Any] = {
            "type": "req",
            "id": connect_id,
            "method": "connect",
            "params": {
                "minProtocol": 3,
                "maxProtocol": 3,
                "client": {
                    "id": "superred",
                    "version": "0.1.0",
                    "platform": "linux",
                    "mode": "operator",
                },
                "role": "operator",
                "scopes": ["operator.read", "operator.write"],
                "caps": [],
                "commands": [],
                "permissions": {},
                "auth": {"token": self._auth_token},
                "locale": "en-US",
                "userAgent": "superred/0.1.0",
                "device": {
                    "id": self._device_id,
                    "nonce": nonce,
                },
            },
        }

        result = await self._send_request(connect_req, connect_id)
        if not result.get("ok"):
            error = result.get("error", "unknown error")
            raise ConnectionError(f"Gateway connect failed: {error}")

        self._connected = True
        logger.info("Connected to OpenClaw Gateway at %s", self._url)
        return result.get("payload", {})

    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._connected = False
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None

    # ------------------------------------------------------------------
    # Agent RPC
    # ------------------------------------------------------------------

    async def run_agent(
        self,
        message: str,
        *,
        session_key: str = "superred",
        timeout_s: float = 120,
        on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> AgentRunResult:
        """Send a message to the agent and collect the full run result.

        Args:
            message: The user message to send.
            session_key: Session routing key.
            timeout_s: Maximum seconds to wait for the run to complete.
            on_event: Optional async callback invoked for each
                streamed :class:`AgentEvent` as it arrives. Lets
                callers forward live events into the trajectory
                instead of waiting for the aggregated result.

        Returns:
            The aggregated agent run result.
        """
        req_id = self._next_id()
        idempotency_key = uuid.uuid4().hex

        agent_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribe_event("agent", agent_queue)

        try:
            req: dict[str, Any] = {
                "type": "req",
                "id": req_id,
                "method": "agent",
                "params": {
                    "message": message,
                    "sessionKey": session_key,
                    "idempotencyKey": idempotency_key,
                },
            }

            ack = await self._send_request(req, req_id)
            if not ack.get("ok"):
                error = ack.get("error", "unknown")
                return AgentRunResult(
                    run_id="", status="error", assistant_text="", error=str(error),
                )

            ack_payload = ack.get("payload", {})
            run_id = ack_payload.get("runId", req_id)

            return await self._collect_agent_events(
                run_id, agent_queue, timeout_s, on_event,
            )
        finally:
            self._unsubscribe_event("agent", agent_queue)

    async def _collect_agent_events(
        self,
        run_id: str,
        queue: asyncio.Queue[dict[str, Any]],
        timeout_s: float,
        on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> AgentRunResult:
        """Consume agent stream events until the run completes.

        NOTE: the exact ``agent`` event payload shape (the ``stream``
        discriminator and ``assistant``/``tool``/``lifecycle`` field
        names below) must be pinned against the gateway-protocol schema
        for the OpenClaw version under test
        (``packages/gateway-protocol/src/schema.ts``). Parsing is kept
        defensive (missing fields tolerated) until that is verified
        against a live gateway.
        """
        assistant_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        events: list[AgentEvent] = []
        status = "running"
        error: str | None = None

        deadline = asyncio.get_event_loop().time() + timeout_s

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                status = "timeout"
                break

            try:
                raw = await asyncio.wait_for(queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                status = "timeout"
                break

            payload = raw.get("payload", {})
            stream = payload.get("stream", "")
            event = AgentEvent(stream=stream, payload=payload, raw=raw)
            events.append(event)

            if on_event is not None:
                try:
                    await on_event(event)
                except Exception:
                    logger.exception("on_event callback raised; continuing")

            if stream == "assistant":
                text = payload.get("text", "")
                if text:
                    assistant_parts.append(text)

            elif stream == "tool":
                tool_calls.append(payload)

            elif stream == "lifecycle":
                phase = payload.get("phase", "")
                if phase == "end":
                    status = "ok"
                    break
                elif phase == "error":
                    status = "error"
                    error = payload.get("error", "agent error")
                    break

        # The agent RPC may also send a final response after lifecycle:end.
        # We drain remaining events briefly to capture it.
        try:
            while True:
                raw = await asyncio.wait_for(queue.get(), timeout=0.5)
                payload = raw.get("payload", {})
                stream = payload.get("stream", "")
                if stream == "assistant":
                    text = payload.get("text", "")
                    if text:
                        assistant_parts.append(text)
        except asyncio.TimeoutError:
            pass

        return AgentRunResult(
            run_id=run_id,
            status=status,
            assistant_text="".join(assistant_parts),
            tool_calls=tool_calls,
            events=events,
            error=error,
        )

    # ------------------------------------------------------------------
    # Generic RPC helpers
    # ------------------------------------------------------------------

    async def rpc(
        self, method: str, params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a generic RPC request and return the response payload.

        Args:
            method: The RPC method name (e.g. ``"health"``, ``"tools.catalog"``).
            params: Optional parameters dict.

        Returns:
            The response payload dict, or an error dict.
        """
        req_id = self._next_id()
        req: dict[str, Any] = {
            "type": "req",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            req["params"] = params
        result = await self._send_request(req, req_id)
        if result.get("ok"):
            return result.get("payload", {})
        return {"error": result.get("error", "unknown")}

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def reset_session(self, session_key: str = "superred") -> None:
        """Reset (clear) a session on the Gateway."""
        await self.rpc("sessions.reset", {"sessionKey": session_key})

    async def get_session_history(
        self, session_key: str = "superred",
    ) -> list[dict[str, Any]]:
        """Retrieve chat history for a session."""
        result = await self.rpc("chat.history", {"sessionKey": session_key})
        if isinstance(result, dict) and "error" in result:
            return []
        if isinstance(result, list):
            return result
        return result.get("messages", [])

    # ------------------------------------------------------------------
    # Internal transport
    # ------------------------------------------------------------------

    def _next_id(self) -> str:
        return uuid.uuid4().hex[:16]

    async def _send_request(
        self, req: dict[str, Any], req_id: str,
    ) -> dict[str, Any]:
        """Send a request frame and wait for the matching response."""
        if not self._ws:
            raise ConnectionError("Not connected")

        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        await self._ws.send(json.dumps(req))
        try:
            return await asyncio.wait_for(future, timeout=30)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            return {"ok": False, "error": "request timeout"}

    async def _wait_for_event(
        self, event_name: str, timeout: float = 10,
    ) -> dict[str, Any]:
        """Wait for a specific server-push event."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribe_event(event_name, queue)
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        finally:
            self._unsubscribe_event(event_name, queue)

    def _subscribe_event(
        self, event_name: str, queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        self._event_listeners.setdefault(event_name, []).append(queue)

    def _unsubscribe_event(
        self, event_name: str, queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        listeners = self._event_listeners.get(event_name, [])
        if queue in listeners:
            listeners.remove(queue)

    async def _read_loop(self) -> None:
        """Background task that reads frames and dispatches them."""
        assert self._ws is not None
        try:
            async for raw_msg in self._ws:
                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON frame from Gateway: %s", raw_msg[:200])
                    continue

                msg_type = msg.get("type")

                if msg_type == "res":
                    req_id = msg.get("id")
                    future = self._pending.pop(req_id, None)
                    if future and not future.done():
                        future.set_result(msg)

                elif msg_type == "event":
                    event_name = msg.get("event", "")
                    for queue in self._event_listeners.get(event_name, []):
                        queue.put_nowait(msg)
                    # Also dispatch to wildcard listeners
                    for queue in self._event_listeners.get("*", []):
                        queue.put_nowait(msg)

        except websockets.exceptions.ConnectionClosed:
            logger.info("Gateway WebSocket closed")
        except asyncio.CancelledError:
            pass
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("WebSocket closed"))
            self._pending.clear()
