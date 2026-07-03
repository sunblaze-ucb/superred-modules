"""Async WebSocket client for the OpenClaw Gateway protocol.

Handles the connect handshake, request/response dispatch, and event
streaming required to drive the Gateway from Python.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
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
        use_device_identity: Sign connect with Ed25519 device identity (required
            for remote/Docker gateways).
        device_identity_path: Path to ``identity/device.json`` in the gateway
            state dir (paired device must be pre-seeded there).
    """

    def __init__(
        self,
        gateway_url: str,
        auth_token: str,
        device_id: str | None = None,
        use_device_identity: bool = False,
        device_identity_path: str | None = None,
    ) -> None:
        self._url = gateway_url
        self._auth_token = auth_token
        self._device_id = device_id or f"superred-{uuid.uuid4().hex[:12]}"
        self._use_device_identity = use_device_identity
        self._device_identity_path = device_identity_path
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

        challenge = await self._wait_for_event("connect.challenge", timeout=15)
        challenge_nonce = challenge.get("payload", {}).get("nonce", "")

        scopes = ["operator.read", "operator.write", "operator.admin"]
        connect_id = self._next_id()

        if self._use_device_identity:
            if not self._device_identity_path:
                raise ConnectionError(
                    "use_device_identity=True requires device_identity_path",
                )
            from pathlib import Path

            from openclaw_target.device_identity import (
                OPERATOR_SCOPES,
                build_device_auth_payload_v3,
                load_or_create_device_identity,
                public_key_raw_base64url,
                sign_device_payload,
            )

            identity = load_or_create_device_identity(
                Path(self._device_identity_path),
            )
            signed_at_ms = int(time.time() * 1000)
            payload = build_device_auth_payload_v3(
                device_id=identity.device_id,
                client_id="cli",
                client_mode="cli",
                role="operator",
                scopes=list(OPERATOR_SCOPES),
                signed_at_ms=signed_at_ms,
                token=self._auth_token,
                nonce=challenge_nonce,
                platform="linux",
            )
            connect_params: dict[str, Any] = {
                "minProtocol": 3,
                "maxProtocol": 4,
                "client": {
                    "id": "cli",
                    "version": "0.1.0",
                    "platform": "linux",
                    "mode": "cli",
                },
                "role": "operator",
                "scopes": list(scopes),
                "caps": [],
                "commands": [],
                "permissions": {},
                "auth": {"token": self._auth_token},
                "device": {
                    "id": identity.device_id,
                    "publicKey": public_key_raw_base64url(identity.public_key_pem),
                    "signature": sign_device_payload(identity.private_key_pem, payload),
                    "signedAt": signed_at_ms,
                    "nonce": challenge_nonce,
                },
                "locale": "en-US",
                "userAgent": "superred/0.1.0",
            }
        else:
            # Direct-local backend path: loopback + gateway-client/backend only.
            connect_params = {
                "minProtocol": 3,
                "maxProtocol": 4,
                "client": {
                    "id": "gateway-client",
                    "version": "0.1.0",
                    "platform": "linux",
                    "mode": "backend",
                },
                "role": "operator",
                "scopes": scopes,
                "caps": [],
                "commands": [],
                "permissions": {},
                "auth": {"token": self._auth_token},
                "locale": "en-US",
                "userAgent": "superred/0.1.0",
            }

        connect_req: dict[str, Any] = {
            "type": "req",
            "id": connect_id,
            "method": "connect",
            "params": connect_params,
        }

        result: dict[str, Any] = {}
        deadline = time.time() + 30.0
        while time.time() < deadline:
            connect_id = self._next_id()
            connect_req["id"] = connect_id
            result = await self._send_request(connect_req, connect_id)
            if result.get("ok"):
                break
            error = result.get("error")
            if isinstance(error, dict) and error.get("retryable") and error.get("code") == "UNAVAILABLE":
                retry_ms = int(error.get("retryAfterMs") or 500)
                await asyncio.sleep(retry_ms / 1000.0)
                continue
            break

        if not result.get("ok"):
            error = result.get("error", "unknown error")
            raise ConnectionError(f"Gateway connect failed: {error}")

        hello = result.get("payload", {})
        granted = hello.get("auth", {}).get("scopes") or hello.get("scopes") or []
        if self._use_device_identity and scopes and not granted:
            raise ConnectionError(
                f"Gateway granted empty scopes {granted!r}; "
                "device pairing or identity may be misconfigured",
            )

        self._connected = True
        logger.info("Connected to OpenClaw Gateway at %s", self._url)
        return hello

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
        timeout_s: float = 600,
        on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> AgentRunResult:
        """Send a message to the agent and collect the run result.

        Uses the verified two-stage agent flow: the ``agent`` RPC returns
        an immediate ``status:"accepted"`` ack carrying a ``runId``; we then
        block on ``agent.wait`` for the terminal result while forwarding
        live ``chat`` (assistant text) and ``session.tool`` (tool call)
        events. Final assistant text comes from accumulated ``chat`` deltas,
        falling back to ``chat.history``.

        Grounding (gateway/protocol): agent runs are two-stage (accepted ack
        then terminal); ``agent.wait`` returns the terminal snapshot;
        ``chat`` events carry ``deltaText`` with ``message`` as the
        cumulative snapshot in protocol v4 (``replace`` marks a non-prefix
        replacement); ``chat.history`` returns the session messages. The
        per-event field set is documented; exact nested shapes ultimately
        come from the generated protocol schema.

        Args:
            message: The user message to send.
            session_key: Session routing key.
            timeout_s: Maximum seconds to wait for the run to complete.
            on_event: Optional async callback invoked for each live
                :class:`AgentEvent` so callers can forward events into the
                trajectory as they arrive.
        """
        req_id = self._next_id()
        idempotency_key = uuid.uuid4().hex

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribe_event("chat", queue)
        self._subscribe_event("session.tool", queue)

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
                return AgentRunResult(
                    run_id="", status="error", assistant_text="",
                    error=str(ack.get("error", "unknown")),
                )

            run_id = ack.get("payload", {}).get("runId", req_id)
            return await self._collect_run(
                run_id, session_key, queue, timeout_s, on_event,
            )
        finally:
            self._unsubscribe_event("chat", queue)
            self._unsubscribe_event("session.tool", queue)

    @staticmethod
    def _to_agent_event(raw: dict[str, Any]) -> AgentEvent:
        family = raw.get("event", "")
        payload = raw.get("payload", {})
        if family == "session.tool":
            stream = "tool"
        elif family == "chat":
            stream = "chat"
        else:
            stream = family
        return AgentEvent(stream=stream, payload=payload, raw=raw)

    async def _collect_run(
        self,
        run_id: str,
        session_key: str,
        queue: asyncio.Queue[dict[str, Any]],
        timeout_s: float,
        on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> AgentRunResult:
        """Forward live events while ``agent.wait`` resolves the run."""
        assistant_parts: list[str] = []
        snapshot_text = ""
        tool_calls: list[dict[str, Any]] = []
        events: list[AgentEvent] = []
        status = "running"
        error: str | None = None

        # agent.wait blocks until the run reaches a terminal state, which can
        # take far longer than a normal RPC. Disable the per-request timeout
        # (timeout=None) and let _collect_run's own ``deadline`` govern it,
        # otherwise the 30s RPC cap would abort every run longer than 30s.
        wait_task: asyncio.Future[dict[str, Any]] = asyncio.ensure_future(
            self.rpc("agent.wait", {"runId": run_id}, timeout=None),
        )
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_s

        def absorb(event: AgentEvent) -> None:
            nonlocal snapshot_text
            events.append(event)
            if event.stream == "chat":
                delta = event.payload.get("deltaText", "")
                if delta:
                    if event.payload.get("replace"):
                        assistant_parts.clear()
                    assistant_parts.append(delta)
                snap = event.payload.get("message")
                if isinstance(snap, str) and snap:
                    snapshot_text = snap
            elif event.stream == "tool":
                tool_calls.append(event.payload)

        while not wait_task.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                status = "timeout"
                break
            try:
                raw = await asyncio.wait_for(queue.get(), timeout=min(remaining, 0.2))
            except asyncio.TimeoutError:
                continue
            event = self._to_agent_event(raw)
            if on_event is not None:
                try:
                    await on_event(event)
                except Exception:
                    logger.exception("on_event callback raised; continuing")
            absorb(event)

        if status == "timeout":
            wait_task.cancel()
        else:
            try:
                terminal = await asyncio.wait_for(
                    wait_task, timeout=max(0.0, deadline - loop.time()) or 5.0,
                )
                if isinstance(terminal, dict) and terminal.get("error"):
                    status, error = "error", str(terminal["error"])
                else:
                    status = "ok"
            except asyncio.TimeoutError:
                status = "timeout"
                wait_task.cancel()

        # Drain any events already delivered after the run resolved.
        while not queue.empty():
            absorb(self._to_agent_event(queue.get_nowait()))

        assistant_text = "".join(assistant_parts) or snapshot_text
        if not assistant_text and status == "ok":
            assistant_text = await self._last_assistant_from_history(session_key)

        return AgentRunResult(
            run_id=run_id,
            status=status,
            assistant_text=assistant_text,
            tool_calls=tool_calls,
            events=events,
            error=error,
        )

    async def _last_assistant_from_history(self, session_key: str) -> str:
        try:
            history = await self.get_session_history(session_key)
        except Exception:
            return ""
        for msg in reversed(history):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    # ------------------------------------------------------------------
    # Generic RPC helpers
    # ------------------------------------------------------------------

    async def rpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = 30,
    ) -> dict[str, Any]:
        """Send a generic RPC request and return the response payload.

        Args:
            method: The RPC method name (e.g. ``"health"``, ``"tools.catalog"``).
            params: Optional parameters dict.
            timeout: Per-request timeout in seconds, or ``None`` to wait
                indefinitely (used for long-running calls like ``agent.wait``
                whose duration is governed by a higher-level deadline).

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
        result = await self._send_request(req, req_id, timeout=timeout)
        if result.get("ok"):
            return result.get("payload", {})
        return {"error": result.get("error", "unknown")}

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def reset_session(self, session_key: str = "superred") -> None:
        """Reset (clear) a session on the Gateway.

        The ``sessions.*`` namespace keys sessions by ``key`` (verified against
        openclaw/openclaw ``src/acp/session-mapper.ts`` and
        ``src/tui/gateway-chat.ts``), unlike ``agent`` / ``chat.history`` which
        use ``sessionKey``; sending the wrong field makes the gateway's
        ``validateSessionsResetParams`` reject the call. ``rpc`` returns an
        ``{"error": ...}`` dict rather than raising, so inspect it and raise so
        the caller does not silently treat a rejected reset as success.
        """
        result = await self.rpc("sessions.reset", {"key": session_key})
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(f"sessions.reset failed: {result['error']}")

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
        self, req: dict[str, Any], req_id: str, timeout: float | None = 30,
    ) -> dict[str, Any]:
        """Send a request frame and wait for the matching response.

        ``timeout`` is the seconds to wait for the response, or ``None`` to
        wait indefinitely (the caller is responsible for bounding it).
        """
        if not self._ws:
            raise ConnectionError("Not connected")

        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        await self._ws.send(json.dumps(req))
        try:
            if timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            return {"ok": False, "error": "request timeout"}
        except asyncio.CancelledError:
            self._pending.pop(req_id, None)
            raise

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
