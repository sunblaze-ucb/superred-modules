"""Opt-in smoke tests for the Docker managed runtime (real container).

Skipped unless a Docker daemon is reachable. Exercises the issues called out
in review: operator scopes over a published port, injection plugin manifest,
and a full agent turn via stub LLM.

Run explicitly::

    pytest test_docker_smoke.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web

from openclaw_target.device_identity import OPERATOR_SCOPES
from openclaw_target.docker_runtime import DEFAULT_DOCKER_IMAGE, OpenClawDockerRuntime
from openclaw_target.target import _plugin_dir
from openclaw_target.ws_client import OpenClawWSClient


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
