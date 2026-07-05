"""Opt-in smoke tests for the Docker managed runtime (real container).

Skipped unless a Docker daemon is reachable. Exercises the issues called out
in review: operator scopes over a published port, injection plugin manifest,
and a full agent turn via stub LLM.

Run explicitly::

    pytest test_docker_smoke.py -v
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from openclaw_target.device_identity import OPERATOR_SCOPES
from openclaw_target.docker_runtime import DEFAULT_DOCKER_IMAGE, OpenClawDockerRuntime
from openclaw_target.injection_server import InjectionServer
from openclaw_target.target import FILE_CONTENT_CTRL, OpenClawTarget, USER_MESSAGE_CTRL, _plugin_dir
from openclaw_target.ws_client import OpenClawWSClient

from superred.core.types.events import (
    ControllableInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
)


def _docker_daemon_ready() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(
        ["docker", "info"],
        capture_output=True,
        check=False,
    ).returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_daemon_ready(),
    reason="Docker daemon unavailable",
)


@asynccontextmanager
async def _stub_llm_server(
    *,
    reply: str = "Docker stub LLM reply.",
) -> AsyncIterator[str]:
    """Start a stub LLM server and yield the URL the *container* reaches it on.

    Binds ``0.0.0.0`` (not just loopback) and returns a ``host.docker.internal``
    URL — a containerised gateway resolves ``127.0.0.1`` to itself, not the
    host, so a loopback-only stub is unreachable from inside the container.
    """

    async def completions(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "choices": [{"message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 0)  # noqa: S104 - needed for container reachability
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://host.docker.internal:{port}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def _stub_llm_upstream_for_host_proxy(
    *,
    reply: str = "Docker Target stub LLM reply.",
) -> AsyncIterator[str]:
    """Stub upstream for :class:`OpenClawTarget`'s host-side LLM proxy.

    With ``managed_runtime="docker"`` the container reaches the proxy on
    ``host.docker.internal``; the proxy (on the host) forwards to this
    loopback-only upstream.
    """

    async def completions(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "choices": [{"message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def _stub_tool_calling_upstream_for_host_proxy(
    *,
    tool_name: str,
    tool_arguments: dict[str, Any],
    final_reply: str = "Docker Target tool loop complete.",
) -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    """Tool-calling stub upstream for the host-side LLM proxy (see above)."""
    requests: list[dict[str, Any]] = []

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(body)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_arguments),
                        },
                    },
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": final_reply}
            finish_reason = "stop"
        return web.json_response(
            {
                "id": f"chatcmpl-docker-target-stub-{len(requests)}",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason},
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}", requests
    finally:
        await runner.cleanup()


def _docker_image() -> str:
    return os.environ.get("OPENCLAW_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)


def _docker_target(**kwargs: Any) -> OpenClawTarget:
    """Build an :class:`OpenClawTarget` on the containerised managed runtime."""
    return OpenClawTarget(
        managed=True,
        managed_runtime="docker",
        managed_kwargs={"image": _docker_image()},
        **kwargs,
    )


def _passthrough_send_event(
    user_message: str,
    hook_calls: list[tuple[str, dict[str, Any]]],
    *,
    injection_marker: str = "DOCKER-TARGET-INJECTED-88",
):
    async def send_event(event: object) -> ControllableInjection:
        controllable = getattr(event, "controllable")
        if isinstance(event, ControllablePostCallEvent) and controllable is FILE_CONTENT_CTRL:
            hook_calls.append((controllable.name, json.loads(event.request)))
            return ControllableInjection(
                event=event,  # type: ignore[arg-type]
                controllable=controllable,
                value=injection_marker,
            )
        if isinstance(event, ControllablePreCallEvent) and controllable is USER_MESSAGE_CTRL:
            return ControllableInjection(
                event=event,  # type: ignore[arg-type]
                controllable=controllable,
                value=user_message,
            )
        return ControllableInjection(
            event=event, controllable=controllable, value="",  # type: ignore[arg-type]
        )

    return send_event


@asynccontextmanager
async def _stub_tool_calling_llm_server(
    *,
    tool_name: str,
    tool_arguments: dict[str, Any],
    final_reply: str = "Docker tool loop complete.",
) -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    """Docker counterpart of the live-test stub: calls one tool on turn 1,
    then finishes on turn 2, so the *containerised* gateway drives a real
    tool-calling loop through the real (in-container) plugin. Binds
    ``0.0.0.0`` and yields a ``host.docker.internal`` URL for container
    reachability (see ``_stub_llm_server`` above)."""
    requests: list[dict[str, Any]] = []

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(body)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_arguments),
                        },
                    },
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": final_reply}
            finish_reason = "stop"
        return web.json_response(
            {
                "id": f"chatcmpl-docker-stub-{len(requests)}",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason},
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 0)  # noqa: S104 - needed for container reachability
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://host.docker.internal:{port}", requests
    finally:
        await runner.cleanup()


def _docker_client(rt: OpenClawDockerRuntime) -> OpenClawWSClient:
    return OpenClawWSClient(
        gateway_url=rt.gateway_url,
        auth_token=rt.auth_token or "",
        use_device_identity=rt.use_device_identity,
        device_identity_path=rt.device_identity_path,
    )


@pytest.mark.asyncio
async def test_docker_connect_grants_operator_scopes() -> None:
    """Remote connect must receive write/admin scopes (not empty)."""
    image = os.environ.get("OPENCLAW_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)
    rt = OpenClawDockerRuntime(image=image, model_id="openai/gpt-4o-mini")
    await rt.start()
    try:
        client = _docker_client(rt)
        hello = await client.connect()
        granted = hello.get("auth", {}).get("scopes") or hello.get("scopes") or []
        for scope in OPERATOR_SCOPES:
            assert scope in granted, f"missing {scope} in {granted}"
        await client.close()
    finally:
        await rt.stop()


@pytest.mark.asyncio
async def test_docker_gateway_rpc_and_agent_run() -> None:
    """Real container: RPCs + agent turn with stub upstream."""
    image = os.environ.get("OPENCLAW_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)
    async with _stub_llm_server() as stub_url:
        rt = OpenClawDockerRuntime(
            image=image,
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
        )
        await rt.start()
        try:
            client = _docker_client(rt)
            await client.connect()
            catalog = await client.rpc("tools.catalog")
            assert isinstance(catalog, dict)

            await client.rpc(
                "agents.files.set",
                {
                    "agentId": "main",
                    "name": "AGENTS.md",
                    "content": "# docker smoke",
                },
            )
            await client.reset_session("docker-smoke")

            result = await client.run_agent(
                "Say hello",
                session_key="docker-smoke",
                timeout_s=120,
            )
            assert result.status == "ok"
            assert result.error is None
            assert "Docker stub LLM reply." in result.assistant_text
            await client.close()
        finally:
            await rt.stop()


@pytest.mark.asyncio
async def test_docker_gateway_starts_with_injection_plugin() -> None:
    """Gateway boots when the superred injection extension is installed."""
    image = os.environ.get("OPENCLAW_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)
    plugin = _plugin_dir()
    manifest = plugin / "openclaw.plugin.json"
    assert manifest.is_file(), "plugin manifest required for injection runs"

    rt = OpenClawDockerRuntime(
        image=image,
        model_id="openai/gpt-4o-mini",
        plugin_dir=str(plugin),
    )
    await rt.start()
    try:
        ext = Path(rt.device_identity_path or "").parent.parent / "extensions" / "superred-injection"
        assert (ext / "openclaw.plugin.json").is_file()
        assert (ext / "index.js").is_file()

        client = _docker_client(rt)
        await client.connect()
        await client.close()
    finally:
        await rt.stop()


@pytest.mark.asyncio
async def test_docker_tool_injection_round_trip_through_real_plugin() -> None:
    """The injection plugin's hooks fire for real *inside the container*.

    Same shape as ``test_openclaw_live.py``'s
    ``test_live_tool_injection_round_trip_through_real_plugin``, but against
    a real containerised gateway: the in-container Node plugin POSTs to a
    host-side :class:`InjectionServer` over ``host.docker.internal``, and the
    persisted tool result carries the injected content into the next prompt
    submission in the same session. Proves the Docker path isn't just
    "the manifest loads without crashing" (``test_docker_gateway_starts_with_
    injection_plugin``) but that the actual hook round trip works end to end
    from inside the container.
    """
    image = os.environ.get("OPENCLAW_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)
    plugin = _plugin_dir()
    callback_token = secrets.token_urlsafe(16)
    hook_calls: list[dict[str, Any]] = []

    async def hook_handler(
        hook_type: str,
        tool_name: str,
        params: dict[str, Any],
        tool_call_id: str,
        result: Any,
    ) -> dict[str, Any] | None:
        if hook_type == "before_tool_call":
            hook_calls.append(
                {"tool": tool_name, "params": params, "toolCallId": tool_call_id},
            )
            return {"toolResult": "DOCKER-INJECTED-BY-SUPERRED-77"}
        return None

    injection = InjectionServer(
        handler=hook_handler, host="0.0.0.0", port=0, auth_token=callback_token,  # noqa: S104
    )
    await injection.start()

    async with _stub_tool_calling_llm_server(
        tool_name="read",
        tool_arguments={"path": "USER.md"},
    ) as (stub_url, requests):
        rt = OpenClawDockerRuntime(
            image=image,
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            plugin_dir=str(plugin),
            callback_url=f"http://host.docker.internal:{injection.actual_port}",
            callback_token=callback_token,
        )
        await rt.start()
        try:
            client = _docker_client(rt)
            await client.connect()
            await client.rpc(
                "agents.files.set",
                {
                    "agentId": "main",
                    "name": "USER.md",
                    "content": "DOCKER-ORIGINAL-FILE-CONTENT-NOT-INJECTED",
                },
            )
            await client.reset_session("docker-injection")

            result = await client.run_agent(
                "Read USER.md and tell me what it says.",
                session_key="docker-injection",
                timeout_s=120,
            )
            assert result.status == "ok"
            assert result.error is None

            # 1. The real in-container plugin actually POSTed to the
            # host-side injection server for a real tool call.
            assert len(hook_calls) == 1, f"plugin hook never fired: {hook_calls}"
            assert hook_calls[0]["tool"] == "read"
            assert hook_calls[0]["params"].get("path") == "USER.md"
            assert len(requests) >= 2

            # 2. A follow-up prompt submission in the same session loads the
            # persisted (poisoned) transcript - the toolResult message now
            # carries the injected content, not the real file content.
            second_result = await client.run_agent(
                "What did USER.md say?",
                session_key="docker-injection",
                timeout_s=120,
            )
            assert second_result.status == "ok"
            assert len(requests) >= 3
            next_prompt_messages = requests[-1]["messages"]
            tool_messages = [m for m in next_prompt_messages if m.get("role") == "tool"]
            assert tool_messages, f"expected persisted tool message: {next_prompt_messages}"
            assert any(
                "DOCKER-INJECTED-BY-SUPERRED-77" in str(m.get("content"))
                for m in tool_messages
            )
            assert all(
                "DOCKER-ORIGINAL-FILE-CONTENT-NOT-INJECTED" not in str(m.get("content"))
                for m in tool_messages
            )

            await client.close()
        finally:
            await rt.stop()
            await injection.stop()


@pytest.mark.asyncio
async def test_docker_openclaw_target_managed_run_with_stub_llm() -> None:
    """``OpenClawTarget(managed_runtime=\"docker\")`` end to end via the Target.

    Unlike the other tests here (which call :class:`OpenClawDockerRuntime`
    directly), this exercises the integrated path: Target starts the
    host-side LLM proxy + optional injection server, materialises state,
    launches the container, connects over WS, and completes a managed run.
    """
    async with _stub_llm_upstream_for_host_proxy() as stub_url:
        target = _docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
        )
        try:
            await target.warmup_static_observables()
            assert target._runtime is not None
            assert target._llm_proxy is not None

            async def send_event(event: object) -> ControllableInjection:
                controllable = getattr(event, "controllable")
                value = (
                    "Say hello from the Docker Target path."
                    if controllable is USER_MESSAGE_CTRL
                    else ""
                )
                return ControllableInjection(
                    event=event, controllable=controllable, value=value,  # type: ignore[arg-type]
                )

            await target.run(lambda _e: None, send_event)

            response = target.query("last_response")
            assert response is not None
            assert "Docker Target stub LLM reply." in response
            assert target._llm_proxy.records, "proxy should record the upstream call"
            user_messages = [
                m.get("content", "")
                for m in target._llm_proxy.records[-1].request_messages
                if m.get("role") == "user"
            ]
            assert any(
                "Say hello from the Docker Target path." in str(content)
                for content in user_messages
            ), user_messages
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_openclaw_target_tool_injection_round_trip() -> None:
    """Tool injection through ``OpenClawTarget``'s Docker managed runtime.

    Same proof as ``test_docker_tool_injection_round_trip_through_real_plugin``,
    but routed through :class:`OpenClawTarget` so the container reaches the
    host-side injection server and LLM proxy URLs that the Target constructs.
    """
    async with _stub_tool_calling_upstream_for_host_proxy(
        tool_name="read",
        tool_arguments={"path": "USER.md"},
    ) as (stub_url, requests):
        target = _docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            enable_tool_injection=True,
        )
        try:
            await target.warmup_static_observables()
            target.set_config(
                "workspace_files",
                json.dumps({"USER.md": "DOCKER-TARGET-ORIGINAL-NOT-INJECTED"}),
            )

            hook_calls: list[tuple[str, dict[str, Any]]] = []
            send_event = _passthrough_send_event(
                "Read USER.md and tell me what it says.",
                hook_calls,
            )
            await target.run(lambda _e: None, send_event)

            assert len(hook_calls) == 1, f"plugin hook never fired: {hook_calls}"
            assert hook_calls[0][0] == "file_content"
            assert hook_calls[0][1]["tool"] == "read"
            assert len(requests) >= 2

            requests_before = len(requests)
            hook_calls.clear()
            send_event_2 = _passthrough_send_event("What did USER.md say?", hook_calls)
            await target.run(lambda _e: None, send_event_2)

            assert len(requests) > requests_before
            tool_messages = [
                m
                for m in requests[requests_before]["messages"]
                if m.get("role") == "tool"
            ]
            assert tool_messages, (
                f"expected persisted tool message: {requests[requests_before]['messages']}"
            )
            assert any(
                "DOCKER-TARGET-INJECTED-88" in str(m.get("content"))
                for m in tool_messages
            )
            assert all(
                "DOCKER-TARGET-ORIGINAL-NOT-INJECTED" not in str(m.get("content"))
                for m in tool_messages
            )
        finally:
            await target.teardown()


DEFAULT_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/v1"
DEFAULT_GEMINI_MODEL = "google/gemini-2.5-flash"
_DOCKER_PROVIDER_TIMEOUT_S = 240


def _gemini_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key or None


def _provider_model() -> str:
    return os.environ.get("OPENCLAW_PROVIDER_MODEL", DEFAULT_GEMINI_MODEL)


def _provider_base_url() -> str:
    return os.environ.get("OPENCLAW_PROVIDER_BASE_URL", DEFAULT_GEMINI_BASE)


@pytest.mark.provider
@pytest.mark.skipif(_gemini_api_key() is None, reason="GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_docker_openclaw_target_real_gemini_turn() -> None:
    """``OpenClawTarget(managed_runtime=\"docker\")`` with a real Gemini upstream."""
    key = _gemini_api_key()
    assert key is not None
    target = _docker_target(
        model_id=_provider_model(),
        provider_base_url=_provider_base_url(),
        provider_api_key=key,
        agent_timeout_s=_DOCKER_PROVIDER_TIMEOUT_S,
    )
    try:
        await target.warmup_static_observables()

        async def send_event(event: object) -> ControllableInjection:
            controllable = getattr(event, "controllable")
            value = (
                "What is 17 + 25? Reply with only the number."
                if controllable is USER_MESSAGE_CTRL
                else ""
            )
            return ControllableInjection(
                event=event, controllable=controllable, value=value,  # type: ignore[arg-type]
            )

        await target.run(lambda _e: None, send_event)

        response = target.query("last_response") or ""
        assert re.search(r"\b42\b", response), response
        assert target._llm_proxy is not None
        assert target._llm_proxy.records, "proxy should record the upstream call"
    finally:
        await target.teardown()
