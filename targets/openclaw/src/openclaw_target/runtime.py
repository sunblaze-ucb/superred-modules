"""Local OpenClaw Gateway process lifecycle.

OpenClaw runs primarily as a *local Node daemon* (the Gateway), reached
over a loopback WebSocket — not as a hosted/containerised service.
(OpenClaw uses Docker only as a tool-execution *sandbox*, not as the
gateway transport.) This manager starts ``openclaw gateway`` as a child
process on a loopback port, points the superred injection extension at
the host callback server, and tears the process down on stop. No
container image, no ``host.docker.internal``, no ``/healthz`` HTTP probe
is involved.

It deliberately does not run the gateway in a container; sandboxed
tool execution (e.g. the Agent's Last Exam virtual-X Docker image) is a
separate, additive concern handled by OpenClaw's own sandbox config.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 18789
_READY_POLL_INTERVAL_S = 0.5
_READY_TIMEOUT_S = 30.0


@dataclass
class OpenClawRuntime:
    """Manages a local OpenClaw Gateway process lifecycle.

    Args:
        openclaw_bin: Path/name of the OpenClaw CLI (default ``openclaw``).
        host: Loopback address to bind (default ``127.0.0.1``).
        host_port: Port to bind the Gateway to.
        provider_api_key: API key for the upstream LLM provider
            (passed as ``OPENCLAW_PROVIDER_KEY``).
        workspace_dir: Agent workspace directory. If ``None`` the CLI
            default is used.
        plugin_dir: Directory containing the superred injection
            extension; installed via ``OPENCLAW_EXTENSIONS_DIR``.
        callback_url: URL of the host injection server, exported to the
            plugin as ``SUPERRED_CALLBACK_URL``.
        tool_policy: Optional tool-profile name. Tool restriction in
            OpenClaw is config (``tools.profile`` / ``tools.allow`` /
            ``agents.<id>.tools.allow``), not a runtime RPC; when set, the
            runtime applies it with ``openclaw config set tools.profile``
            before starting the gateway.
        allow_unconfigured: Pass ``--allow-unconfigured`` so a fresh
            gateway starts without an interactive setup step.
        extra_env: Additional environment variables.
        startup_timeout_s: Max seconds to wait for the port to accept
            connections.

    Note:
        The extension-install directory env (``OPENCLAW_EXTENSIONS_DIR``)
        is gateway-version specific and best-effort. This manager is not
        exercised by the mock-gateway test suite (which connects to an
        unmanaged in-process server).
    """

    openclaw_bin: str = "openclaw"
    host: str = "127.0.0.1"
    host_port: int = _DEFAULT_PORT
    provider_api_key: str = ""
    workspace_dir: str | None = None
    plugin_dir: str | None = None
    callback_url: str | None = None
    tool_policy: str | None = None
    allow_unconfigured: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)
    startup_timeout_s: float = _READY_TIMEOUT_S

    _proc: asyncio.subprocess.Process | None = None
    _auth_token: str | None = None

    @property
    def gateway_url(self) -> str:
        return f"ws://{self.host}:{self.host_port}"

    @property
    def auth_token(self) -> str | None:
        return self._auth_token

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["OPENCLAW_GATEWAY_TOKEN"] = self._auth_token or ""
        if self.provider_api_key:
            env["OPENCLAW_PROVIDER_KEY"] = self.provider_api_key
        if self.workspace_dir:
            env["OPENCLAW_WORKSPACE"] = self.workspace_dir
        if self.plugin_dir:
            env["OPENCLAW_EXTENSIONS_DIR"] = self.plugin_dir
        if self.callback_url:
            env["SUPERRED_CALLBACK_URL"] = self.callback_url
        env.update(self.extra_env)
        return env

    def _build_cmd(self) -> list[str]:
        cmd = [
            self.openclaw_bin, "gateway",
            "--host", self.host,
            "--port", str(self.host_port),
        ]
        if self.allow_unconfigured:
            cmd.append("--allow-unconfigured")
        return cmd

    async def start(self) -> None:
        """Start the local gateway process and wait until it accepts connections."""
        if self._proc is not None:
            logger.warning("Runtime already started (pid %s)", self._proc.pid)
            return

        self._auth_token = secrets.token_hex(24)

        if self.tool_policy:
            await self._apply_tool_profile()

        cmd = self._build_cmd()

        logger.info("Starting OpenClaw gateway: %s", " ".join(cmd))
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=self._build_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await self._wait_ready()

    async def _apply_tool_profile(self) -> None:
        """Restrict the agent's tools by writing the gateway tool profile.

        Tool restriction is gateway config, not a runtime RPC. ``openclaw
        config set tools.profile <name>`` is the documented mechanism; this
        runs it best-effort before the gateway starts so the configured
        policy is not silently dropped.
        """
        cmd = [self.openclaw_bin, "config", "set", "tools.profile", self.tool_policy or ""]
        logger.info("Applying tool profile: %s", " ".join(cmd))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(
                    "Could not set tools.profile=%r: %s",
                    self.tool_policy, stderr.decode().strip(),
                )
        except FileNotFoundError:
            logger.warning("openclaw CLI not found; tool profile not applied")

    async def stop(self) -> None:
        """Terminate the gateway process."""
        proc = self._proc
        self._proc = None
        self._auth_token = None
        if proc is None or proc.returncode is not None:
            return

        logger.info("Stopping OpenClaw gateway (pid %s)", proc.pid)
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("Gateway did not exit; killing (pid %s)", proc.pid)
            proc.kill()
            await proc.wait()

    async def _wait_ready(self) -> None:
        """Poll the loopback port until the gateway accepts a TCP connection."""
        elapsed = 0.0
        while elapsed < self.startup_timeout_s:
            if self._proc is not None and self._proc.returncode is not None:
                stderr = b""
                if self._proc.stderr is not None:
                    stderr = await self._proc.stderr.read()
                raise RuntimeError(
                    "OpenClaw gateway exited during startup "
                    f"(code {self._proc.returncode}): {stderr.decode().strip()}",
                )
            if await self._port_open():
                logger.info("OpenClaw gateway ready after %.1fs", elapsed)
                return
            await asyncio.sleep(_READY_POLL_INTERVAL_S)
            elapsed += _READY_POLL_INTERVAL_S

        await self.stop()
        raise TimeoutError(
            f"OpenClaw gateway not ready after {self.startup_timeout_s}s",
        )

    async def _port_open(self) -> bool:
        try:
            _, writer = await asyncio.open_connection(self.host, self.host_port)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except (ConnectionRefusedError, OSError):
            return False
