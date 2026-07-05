"""OpenClaw Gateway process lifecycle (local Node daemon).

This manager starts ``openclaw gateway`` as a child process on a loopback
port, configures it via a per-instance state dir (``OPENCLAW_STATE_DIR``)
containing a grounded ``openclaw.json`` and the superred injection
extension, and tears it down on stop.

For full-isolation / parallel runs the gateway can instead run inside a
container; see :class:`openclaw_target.docker_runtime.OpenClawDockerRuntime`.
Both runtimes share the configuration and readiness helpers in this module.

Configuration is via OpenClaw's real mechanisms (see :mod:`openclaw_target.config`):
``openclaw.json`` for model/provider routing, ``tools.profile`` and
``plugins.allow``; extensions are installed under ``<stateDir>/extensions``.
The provider base URL (e.g. the superred LLM proxy) is written into
``models.providers.*.baseUrl`` rather than passed as an (ungrounded) env var.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import shutil
import socket
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from openclaw_target.config import (
    DEFAULT_PLUGIN_NAME,
    DEFAULT_PROVIDER_API,
    build_gateway_config,
    materialize_state_dir,
)
from openclaw_target.device_identity import ensure_device_auth_for_state_dir

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 18789
_READY_POLL_INTERVAL_S = 0.5
_READY_TIMEOUT_S = 30.0

# In-container paths (mounted/used by both local and Docker runtimes). The
# Docker image pins these under /home/node; locally we point them at the
# per-instance state dir.
CONTAINER_STATE_DIR = "/home/node/.openclaw"
CONTAINER_WORKSPACE_DIR = "/home/node/.openclaw/workspace"


def free_port() -> int:
    """Reserve a free ephemeral TCP port and return it.

    Binds to port 0, reads the assigned port, then closes the socket. There is
    a small TOCTOU window before the gateway binds it, acceptable for test/eval
    orchestration and far safer than a fixed port under ``concurrency>1``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def http_get_ok(host: str, port: int, path: str = "/healthz") -> bool:
    """Best-effort HTTP/1.0 GET returning ``True`` on a 2xx status.

    Uses raw asyncio streams so the runtime does not depend on aiohttp (an
    optional extra). The gateway serves ``/healthz`` (liveness) and ``/readyz``
    (readiness) on the same port as the WebSocket.
    """
    try:
        reader, writer = await asyncio.open_connection(host, port)
    except (ConnectionRefusedError, OSError):
        return False
    try:
        writer.write(
            f"GET {path} HTTP/1.0\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode(),
        )
        await writer.drain()
        status_line = await reader.readline()
        parts = status_line.decode("latin-1", "replace").split()
        return len(parts) >= 2 and parts[1].startswith("2")
    except OSError:
        return False
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


@dataclass
class OpenClawRuntime:
    """Manages a local OpenClaw Gateway process lifecycle.

    Args:
        openclaw_bin: Path/name of the OpenClaw CLI (default ``openclaw``).
        host: Loopback address used to build the URL and probe readiness
            (default ``127.0.0.1``). The gateway is always started with
            ``--bind loopback``.
        host_port: Port to bind the Gateway to. ``0`` (default) picks a free
            ephemeral port so concurrent instances do not collide.
        model_id: Provider-qualified model id written into ``openclaw.json``.
        provider_base_url: Provider base URL for model calls, written to
            ``models.providers.*.baseUrl``. Set to the superred LLM proxy URL
            to route the gateway's model calls through it.
        provider_api_key: API key for the upstream LLM provider
            (``models.providers.*.apiKey``).
        provider_api: OpenClaw provider API adapter id (``MODEL_APIS``).
        workspace_dir: Agent workspace dir (``agents.defaults.workspace``).
            Defaults to ``<state_dir>/workspace``.
        plugin_dir: Source directory of the superred injection extension,
            copied into ``<state_dir>/extensions/<plugin_name>``.
        plugin_name: Extension id (also added to ``plugins.allow``).
        callback_url: URL of the host injection server, exported to the
            plugin as ``SUPERRED_CALLBACK_URL``.
        callback_token: Bearer token the plugin presents to the callback server,
            exported as ``SUPERRED_CALLBACK_TOKEN``.
        tool_policy: Optional ``tools.profile`` name written to config.
        allow_unconfigured: Pass ``--allow-unconfigured`` so a fresh gateway
            starts without an interactive setup step.
        state_dir: Per-instance state dir (``OPENCLAW_STATE_DIR``). A private
            temp dir is created (and removed on stop) when ``None``.
        extra_env: Additional environment variables.
        startup_timeout_s: Max seconds to wait for ``/healthz``.

    Note:
        This manager is not exercised by the mock-gateway test suite (which
        connects to an unmanaged in-process server); the config/command
        builders are covered by unit tests.
    """

    openclaw_bin: str = "openclaw"
    host: str = "127.0.0.1"
    host_port: int = 0
    bind: str = "loopback"
    model_id: str = ""
    provider_base_url: str | None = None
    provider_api_key: str = ""
    provider_api: str = DEFAULT_PROVIDER_API
    workspace_dir: str | None = None
    plugin_dir: str | None = None
    plugin_name: str = DEFAULT_PLUGIN_NAME
    callback_url: str | None = None
    callback_token: str | None = None
    tool_policy: str | None = None
    allow_unconfigured: bool = True
    state_dir: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    unset_env_keys: tuple[str, ...] = ()
    startup_timeout_s: float = _READY_TIMEOUT_S

    _proc: asyncio.subprocess.Process | None = None
    _auth_token: str | None = None
    _state_path: Path | None = None
    _device_identity_path: Path | None = None
    _owns_state_dir: bool = False

    container_host: str = "127.0.0.1"

    @property
    def gateway_url(self) -> str:
        return f"ws://{self.host}:{self.host_port}"

    @property
    def auth_token(self) -> str | None:
        return self._auth_token

    @property
    def device_identity_path(self) -> str | None:
        """Path to the Ed25519 device identity used for remote gateway connects."""
        return str(self._device_identity_path) if self._device_identity_path else None

    @property
    def use_device_identity(self) -> bool:
        # A loopback-bound gateway reached over 127.0.0.1 qualifies for the
        # direct-local backend path (no device identity). Any non-loopback bind
        # is reached as a "remote" client, so the gateway clears device-less
        # scope requests — those connects must sign with device identity.
        return self.bind != "loopback"

    def _prepare_state_dir(self) -> Path:
        if self.state_dir is not None:
            path = Path(self.state_dir)
            path.mkdir(parents=True, exist_ok=True)
        else:
            path = Path(tempfile.mkdtemp(prefix="superred-openclaw-"))
            self._owns_state_dir = True
        self._state_path = path

        config = build_gateway_config(
            model_id=self.model_id,
            provider_base_url=self.provider_base_url or "",
            provider_api_key=self.provider_api_key,
            provider_api=self.provider_api,
            tool_policy=self.tool_policy or "",
            workspace_dir=self.workspace_dir or str(path / "workspace"),
            plugin_names=[self.plugin_name] if self.plugin_dir else None,
        )
        materialize_state_dir(
            path,
            config=config,
            plugin_src=Path(self.plugin_dir) if self.plugin_dir else None,
            plugin_name=self.plugin_name,
        )
        self._device_identity_path = ensure_device_auth_for_state_dir(path)
        return path

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["OPENCLAW_GATEWAY_TOKEN"] = self._auth_token or ""
        if self._state_path is not None:
            env["OPENCLAW_STATE_DIR"] = str(self._state_path)
            env["OPENCLAW_CONFIG_DIR"] = str(self._state_path)
            env["OPENCLAW_CONFIG_PATH"] = str(self._state_path / "openclaw.json")
            env["OPENCLAW_WORKSPACE_DIR"] = self.workspace_dir or str(
                self._state_path / "workspace",
            )
        if self.callback_url:
            env["SUPERRED_CALLBACK_URL"] = self.callback_url
        if self.callback_token:
            env["SUPERRED_CALLBACK_TOKEN"] = self.callback_token
        env.update(self.extra_env)
        for key in self.unset_env_keys:
            env.pop(key, None)
        return env

    def _build_cmd(self) -> list[str]:
        # Verified flags (src/cli/gateway-cli/run-options.ts): the listener
        # interface is `--bind <loopback|lan|tailnet|auto|custom>` and the port
        # is `--port`. A managed local gateway pins loopback by default (no auth
        # required; non-loopback binds are rejected without a token, which we
        # always set). ``bind="lan"`` is used by remote-path tests so a
        # non-loopback client exercises the same device-identity handshake as a
        # containerised gateway reached over a published port.
        cmd = [
            self.openclaw_bin, "gateway",
            "--bind", self.bind,
            "--port", str(self.host_port),
        ]
        if self.allow_unconfigured:
            cmd.append("--allow-unconfigured")
        return cmd

    async def start(self) -> None:
        """Start the local gateway process and wait until ``/healthz`` is green."""
        if self._proc is not None:
            logger.warning("Runtime already started (pid %s)", self._proc.pid)
            return

        if self.host_port == 0:
            self.host_port = free_port()
        self._auth_token = secrets.token_hex(24)

        self._prepare_state_dir()
        cmd = self._build_cmd()

        logger.info("Starting OpenClaw gateway: %s", " ".join(cmd))
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=self._build_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await self._wait_ready()

    async def stop(self) -> None:
        """Terminate the gateway process and remove any owned state dir."""
        proc = self._proc
        self._proc = None
        self._auth_token = None
        try:
            if proc is not None and proc.returncode is None:
                logger.info("Stopping OpenClaw gateway (pid %s)", proc.pid)
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    logger.warning("Gateway did not exit; killing (pid %s)", proc.pid)
                    proc.kill()
                    await proc.wait()
        finally:
            self._cleanup_state_dir()

    def _cleanup_state_dir(self) -> None:
        if self._owns_state_dir and self._state_path is not None:
            shutil.rmtree(self._state_path, ignore_errors=True)
        self._state_path = None
        self._owns_state_dir = False

    async def _wait_ready(self) -> None:
        """Poll ``/healthz`` until the gateway reports healthy.

        Falls back to a TCP connect probe if the HTTP probe never succeeds but
        the port accepts connections (older gateways without ``/healthz``).
        """
        elapsed = 0.0
        tcp_only_ok = False
        while elapsed < self.startup_timeout_s:
            if self._proc is not None and self._proc.returncode is not None:
                stderr = b""
                if self._proc.stderr is not None:
                    stderr = await self._proc.stderr.read()
                raise RuntimeError(
                    "OpenClaw gateway exited during startup "
                    f"(code {self._proc.returncode}): {stderr.decode().strip()}",
                )
            if await http_get_ok(self.host, self.host_port, "/healthz"):
                if await http_get_ok(self.host, self.host_port, "/readyz"):
                    logger.info("OpenClaw gateway ready after %.1fs", elapsed)
                    return
                logger.debug("Gateway /healthz ok but /readyz not yet ready")
            tcp_only_ok = await self._port_open()
            await asyncio.sleep(_READY_POLL_INTERVAL_S)
            elapsed += _READY_POLL_INTERVAL_S

        if tcp_only_ok:
            logger.warning("Gateway port open but /healthz never 2xx; proceeding")
            return

        await self.stop()
        raise TimeoutError(
            f"OpenClaw gateway not ready after {self.startup_timeout_s}s",
        )

    async def _port_open(self) -> bool:
        try:
            _, writer = await asyncio.open_connection(self.host, self.host_port)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return True
        except (ConnectionRefusedError, OSError):
            return False
