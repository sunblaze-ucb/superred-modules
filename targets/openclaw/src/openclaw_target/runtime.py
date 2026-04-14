"""Docker lifecycle management for OpenClaw.

Starts and stops an OpenClaw Gateway container so the target
adapter can run without manual setup.  Uses the ``docker`` CLI
via subprocess — no Python Docker SDK required.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "ghcr.io/openclaw/openclaw:latest"
_DEFAULT_PORT = 18789
_CONTAINER_PREFIX = "superred-openclaw"
_HEALTH_POLL_INTERVAL_S = 2.0
_HEALTH_TIMEOUT_S = 60.0


@dataclass
class OpenClawRuntime:
    """Manages an OpenClaw Docker container lifecycle.

    Args:
        image: Container image to use.
        host_port: Host port to bind the Gateway to.
        provider_api_key: API key for the upstream LLM provider
            (passed as ``OPENCLAW_PROVIDER_KEY`` env var).
        workspace_dir: Host directory to mount as the agent workspace.
            If ``None``, a temporary directory inside the container is used.
        extra_env: Additional environment variables for the container.
    """

    image: str = _DEFAULT_IMAGE
    host_port: int = _DEFAULT_PORT
    provider_api_key: str = ""
    workspace_dir: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    _container_id: str | None = None
    _auth_token: str | None = None

    @property
    def gateway_url(self) -> str:
        return f"ws://127.0.0.1:{self.host_port}"

    @property
    def auth_token(self) -> str | None:
        return self._auth_token

    async def start(self) -> None:
        """Pull image if needed and start the container."""
        if self._container_id is not None:
            logger.warning("Runtime already started (container %s)", self._container_id)
            return

        self._auth_token = secrets.token_hex(24)
        container_name = f"{_CONTAINER_PREFIX}-{secrets.token_hex(4)}"

        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{self.host_port}:18789",
            "-e", f"OPENCLAW_GATEWAY_TOKEN={self._auth_token}",
        ]

        if self.provider_api_key:
            cmd.extend(["-e", f"OPENCLAW_PROVIDER_KEY={self.provider_api_key}"])

        if self.workspace_dir:
            cmd.extend(["-v", f"{self.workspace_dir}:/home/node/workspace"])

        for key, val in self.extra_env.items():
            cmd.extend(["-e", f"{key}={val}"])

        cmd.append(self.image)

        logger.info("Starting OpenClaw container: %s", " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to start OpenClaw container: {stderr.decode().strip()}"
            )

        self._container_id = stdout.decode().strip()[:12]
        logger.info("Container started: %s", self._container_id)

        await self._wait_healthy()

    async def stop(self) -> None:
        """Stop and remove the container."""
        if self._container_id is None:
            return

        cid = self._container_id
        self._container_id = None
        self._auth_token = None

        logger.info("Stopping OpenClaw container %s", cid)
        proc = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", cid,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()

    async def restart(self) -> None:
        """Tear down and recreate the container for between-task isolation.

        Guarantees a clean filesystem, memory, and session state by
        fully replacing the container. The Gateway URL is preserved
        (same host port) but the auth token is regenerated, so callers
        must reconnect with :attr:`auth_token`.
        """
        logger.info("Restarting OpenClaw container for isolation")
        await self.stop()
        await self.start()

    async def is_healthy(self) -> bool:
        """Check if the Gateway is responding."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", self._container_id or "",
                "curl", "-sf", "http://127.0.0.1:18789/healthz",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception:
            return False

    async def _wait_healthy(self) -> None:
        """Poll until the Gateway is healthy or timeout."""
        elapsed = 0.0
        while elapsed < _HEALTH_TIMEOUT_S:
            if await self.is_healthy():
                logger.info("OpenClaw Gateway healthy after %.1fs", elapsed)
                return
            await asyncio.sleep(_HEALTH_POLL_INTERVAL_S)
            elapsed += _HEALTH_POLL_INTERVAL_S

        raise TimeoutError(
            f"OpenClaw Gateway not healthy after {_HEALTH_TIMEOUT_S}s"
        )
