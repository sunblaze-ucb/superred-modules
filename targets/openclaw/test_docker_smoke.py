"""Opt-in smoke test for the Docker managed runtime.

Skipped unless a Docker daemon is reachable **and** the configured OpenClaw
gateway image exists locally (default ``openclaw:local``). Override the image
with ``OPENCLAW_DOCKER_IMAGE``.

Build the image from an OpenClaw checkout::

    docker build -t openclaw:local .

Run explicitly::

    pytest test_docker_smoke.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from openclaw_target.docker_runtime import OpenClawDockerRuntime
from openclaw_target.ws_client import OpenClawWSClient

_DEFAULT_IMAGE = "openclaw:local"


def _docker_smoke_ready() -> bool:
    if not shutil.which("docker"):
        return False
    if subprocess.run(
        ["docker", "info"],
        capture_output=True,
        check=False,
    ).returncode != 0:
        return False
    image = os.environ.get("OPENCLAW_DOCKER_IMAGE", _DEFAULT_IMAGE)
    return subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
    ).returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_smoke_ready(),
    reason=(
        "Docker daemon unavailable or OpenClaw image missing "
        f"(set OPENCLAW_DOCKER_IMAGE; default {_DEFAULT_IMAGE})"
    ),
)


@pytest.mark.asyncio
async def test_docker_runtime_lifecycle_and_gateway_rpc() -> None:
    """Start a real gateway container, connect, and fetch tools.catalog."""
    image = os.environ.get("OPENCLAW_DOCKER_IMAGE", _DEFAULT_IMAGE)
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
