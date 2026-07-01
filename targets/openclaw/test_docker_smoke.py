"""Opt-in smoke test for the Docker managed runtime.

Skipped unless a Docker daemon is reachable. The runtime auto-pulls the
official OpenClaw gateway image on first start (default
``ghcr.io/openclaw/openclaw:latest``). Override with ``OPENCLAW_DOCKER_IMAGE``.

Run explicitly::

    pytest test_docker_smoke.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from openclaw_target.docker_runtime import DEFAULT_DOCKER_IMAGE, OpenClawDockerRuntime
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


@pytest.mark.asyncio
async def test_docker_runtime_lifecycle_and_gateway_rpc() -> None:
    """Start a real gateway container, connect, and fetch tools.catalog."""
    image = os.environ.get("OPENCLAW_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)
    rt = OpenClawDockerRuntime(
        image=image,
        model_id="openai/gpt-5",
    )
    await rt.start()
    try:
        assert rt.auth_token
        client = OpenClawWSClient(
            gateway_url=rt.gateway_url,
            auth_token=rt.auth_token,
        )
        await client.connect()
        catalog = await client.rpc("tools.catalog")
        assert isinstance(catalog, dict)
        await client.close()
    finally:
        await rt.stop()
