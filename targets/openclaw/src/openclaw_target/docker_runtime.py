"""OpenClaw Gateway lifecycle in a per-instance Docker container.

Runs the whole gateway inside a fresh container for full host isolation and
safe parallelism. Grounded in the real OpenClaw ``Dockerfile`` /
``docker-compose.yml`` and the published release images documented at
https://docs.openclaw.ai/install/docker:

- Image ``ghcr.io/openclaw/openclaw:latest`` by default (official GHCR release;
  Docker Hub mirror: ``openclaw/openclaw:latest``). Override with ``image=`` or
  ``OPENCLAW_DOCKER_IMAGE``; ``openclaw:local`` remains valid for dev builds.
- On ``start()``, the image is pulled automatically when missing locally
  (``auto_pull_image=True``).
- The gateway is launched as ``openclaw gateway --bind lan --port 18789``.
  ``lan`` (0.0.0.0) is required so the host can reach it through a published
  port; OpenClaw *rejects non-loopback binds without auth*, so a gateway token
  is always set (``OPENCLAW_GATEWAY_TOKEN``).
- The host reaches the container via ``-p <dynamicHostPort>:18789``; the
  container reaches host-side services (the injection server and LLM proxy) via
  ``--add-host host.docker.internal:host-gateway`` (works on Linux Docker too).
- State (``openclaw.json`` + ``extensions/``) is materialized in a per-instance
  host dir bind-mounted at ``/home/node/.openclaw``.
- Readiness is the container's own ``GET /healthz`` probe, polled on the
  published host port.

Concurrency safety falls out of this design: dynamic host port, dedicated
container name, and a private state/workspace dir per instance.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from openclaw_target.config import (
    DEFAULT_PLUGIN_NAME,
    DEFAULT_PROVIDER_API,
    build_gateway_config,
    materialize_state_dir,
)
from openclaw_target.runtime import (
    CONTAINER_STATE_DIR,
    CONTAINER_WORKSPACE_DIR,
    free_port,
    http_get_ok,
)

logger = logging.getLogger(__name__)

# Official release image (GHCR primary; Docker Hub mirror: openclaw/openclaw).
DEFAULT_DOCKER_IMAGE = "ghcr.io/openclaw/openclaw:latest"
_DEFAULT_IMAGE = os.environ.get("OPENCLAW_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)
_GATEWAY_CONTAINER_PORT = 18789
_READY_POLL_INTERVAL_S = 0.5
_READY_TIMEOUT_S = 120.0


@dataclass
class OpenClawDockerRuntime:
    """Manages an OpenClaw Gateway running in a dedicated Docker container.

    Mirrors :class:`openclaw_target.runtime.OpenClawRuntime`'s interface
    (``gateway_url``, ``auth_token``, ``container_host``, ``start``/``stop``) so
    the target can swap between local and containerised gateways.

    Args:
        image: Gateway image (default ``ghcr.io/openclaw/openclaw:latest``, or
            ``OPENCLAW_DOCKER_IMAGE`` when set). Official mirrors also include
            ``openclaw/openclaw:<tag>`` on Docker Hub; ``openclaw:local`` for
            dev builds from an OpenClaw checkout.
        docker_bin: Docker CLI (default ``docker``; e.g. ``podman``).
        auto_pull_image: When ``True`` (default), ``start()`` runs
            ``docker pull`` if the configured image is not present locally.
        host: Loopback host used to reach the published port (``127.0.0.1``).
        host_port: Published host port mapped to the container's 18789. ``0``
            (default) picks a free ephemeral port for concurrency.
        container_host: Hostname the *container* uses to reach host services
            (default ``host.docker.internal``). The callback/provider URLs must
            already be expressed in terms of this host.
        model_id / provider_base_url / provider_api_key / provider_api:
            Written into ``openclaw.json`` (provider routing). ``provider_base_url``
            should point at the host LLM proxy via ``container_host``.
        tool_policy: ``tools.profile`` name.
        plugin_dir: Source dir of the injection extension (mounted in).
        plugin_name: Extension id (also ``plugins.allow``).
        callback_url: Injection-server URL the in-container plugin posts to
            (``SUPERRED_CALLBACK_URL``), expressed via ``container_host``.
        callback_token: Bearer token the in-container plugin presents to the
            callback server (``SUPERRED_CALLBACK_TOKEN``).
        state_dir: Host dir to materialize + mount. A private temp dir is used
            (and removed on stop) when ``None``.
        container_name: Container name; a unique one is generated when ``None``.
        extra_run_args: Extra ``docker run`` args (e.g. ``--memory``).
        extra_env: Extra ``-e`` env vars passed to the container.
        remove_on_stop: ``docker rm`` the container on stop (default ``True``).
        startup_timeout_s: Max seconds to wait for ``/healthz``.
    """

    image: str = _DEFAULT_IMAGE
    docker_bin: str = "docker"
    auto_pull_image: bool = True
    host: str = "127.0.0.1"
    host_port: int = 0
    container_host: str = "host.docker.internal"
    model_id: str = ""
    provider_base_url: str | None = None
    provider_api_key: str = ""
    provider_api: str = DEFAULT_PROVIDER_API
    tool_policy: str | None = None
    plugin_dir: str | None = None
    plugin_name: str = DEFAULT_PLUGIN_NAME
    callback_url: str | None = None
    callback_token: str | None = None
    state_dir: str | None = None
    container_name: str | None = None
    extra_run_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)
    remove_on_stop: bool = True
    startup_timeout_s: float = _READY_TIMEOUT_S

    _container_id: str | None = None
    _auth_token: str | None = None
    _state_path: Path | None = None
    _owns_state_dir: bool = False

    @property
    def gateway_url(self) -> str:
        return f"ws://{self.host}:{self.host_port}"

    @property
    def auth_token(self) -> str | None:
        return self._auth_token

    @property
    def container_id(self) -> str | None:
        return self._container_id

    def _prepare_state_dir(self) -> Path:
        if self.state_dir is not None:
            path = Path(self.state_dir)
            path.mkdir(parents=True, exist_ok=True)
        else:
            path = Path(tempfile.mkdtemp(prefix="superred-openclaw-docker-"))
            self._owns_state_dir = True
        self._state_path = path

        # Workspace path in config is the *in-container* path (the gateway reads
        # config inside the container), not the host mount source.
        config = build_gateway_config(
            model_id=self.model_id,
            provider_base_url=self.provider_base_url or "",
            provider_api_key=self.provider_api_key,
            provider_api=self.provider_api,
            tool_policy=self.tool_policy or "",
            workspace_dir=CONTAINER_WORKSPACE_DIR,
            plugin_names=[self.plugin_name] if self.plugin_dir else None,
        )
        materialize_state_dir(
            path,
            config=config,
            plugin_src=Path(self.plugin_dir) if self.plugin_dir else None,
            plugin_name=self.plugin_name,
        )
        return path

    def _build_run_cmd(self) -> list[str]:
        """Build the ``docker run`` argv (pure; depends only on resolved fields)."""
        name = self.container_name or f"superred-openclaw-{secrets.token_hex(6)}"
        self.container_name = name
        cmd = [
            self.docker_bin, "run", "-d",
            "--name", name,
            "-p", f"{self.host_port}:{_GATEWAY_CONTAINER_PORT}",
            "--add-host", "host.docker.internal:host-gateway",
            # Match compose hardening.
            "--cap-drop", "NET_RAW",
            "--cap-drop", "NET_ADMIN",
            "--security-opt", "no-new-privileges:true",
            "--init",
            "-e", f"OPENCLAW_GATEWAY_TOKEN={self._auth_token or ''}",
            "-e", f"OPENCLAW_STATE_DIR={CONTAINER_STATE_DIR}",
            "-e", f"OPENCLAW_CONFIG_DIR={CONTAINER_STATE_DIR}",
            "-e", f"OPENCLAW_CONFIG_PATH={CONTAINER_STATE_DIR}/openclaw.json",
            "-e", f"OPENCLAW_WORKSPACE_DIR={CONTAINER_WORKSPACE_DIR}",
        ]
        if self.callback_url:
            cmd += ["-e", f"SUPERRED_CALLBACK_URL={self.callback_url}"]
        if self.callback_token:
            cmd += ["-e", f"SUPERRED_CALLBACK_TOKEN={self.callback_token}"]
        for key, value in self.extra_env.items():
            cmd += ["-e", f"{key}={value}"]
        if self._state_path is not None:
            cmd += ["-v", f"{self._state_path}:{CONTAINER_STATE_DIR}"]
        cmd += list(self.extra_run_args)
        cmd += [
            self.image,
            "openclaw", "gateway",
            "--bind", "lan",
            "--port", str(_GATEWAY_CONTAINER_PORT),
            "--allow-unconfigured",
        ]
        return cmd

    async def _ensure_image(self) -> None:
        """Ensure the gateway image exists locally, pulling when configured."""
        code, _ = await self._docker(["image", "inspect", self.image], check=False)
        if code == 0:
            return
        if not self.auto_pull_image:
            raise RuntimeError(
                f"Docker image {self.image!r} not found locally and "
                f"auto_pull_image=False. Pull it with: "
                f"{self.docker_bin} pull {self.image}",
            )
        logger.info("Pulling OpenClaw gateway image: %s", self.image)
        await self._docker(["pull", self.image], check=True)

    async def start(self) -> None:
        if self._container_id is not None:
            logger.warning("Docker runtime already started (%s)", self._container_id)
            return

        if self.host_port == 0:
            self.host_port = free_port()
        self._auth_token = secrets.token_hex(24)

        await self._ensure_image()
        self._prepare_state_dir()
        cmd = self._build_run_cmd()

        logger.info("Starting OpenClaw gateway container: %s", " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            self._cleanup_state_dir()
            raise RuntimeError(
                f"docker run failed (code {proc.returncode}): {stderr.decode().strip()}",
            )
        self._container_id = stdout.decode().strip()

        try:
            await self._wait_ready()
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        container = self._container_id
        self._container_id = None
        self._auth_token = None
        try:
            if container is not None:
                await self._docker(["stop", container], check=False)
                if self.remove_on_stop:
                    await self._docker(["rm", "-f", container], check=False)
        finally:
            self._cleanup_state_dir()

    async def _docker(self, args: list[str], *, check: bool) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        code = proc.returncode or 0
        if check and code != 0:
            raise RuntimeError(f"docker {args[0]} failed: {out.decode().strip()}")
        return code, out.decode()

    def _cleanup_state_dir(self) -> None:
        if self._owns_state_dir and self._state_path is not None:
            shutil.rmtree(self._state_path, ignore_errors=True)
        self._state_path = None
        self._owns_state_dir = False

    async def _wait_ready(self) -> None:
        elapsed = 0.0
        while elapsed < self.startup_timeout_s:
            if not await self._container_running():
                logs = await self._container_logs()
                raise RuntimeError(
                    f"OpenClaw gateway container exited during startup: {logs}",
                )
            if await http_get_ok(self.host, self.host_port, "/healthz"):
                logger.info("OpenClaw gateway container healthy after %.1fs", elapsed)
                return
            await asyncio.sleep(_READY_POLL_INTERVAL_S)
            elapsed += _READY_POLL_INTERVAL_S
        raise TimeoutError(
            f"OpenClaw gateway container not ready after {self.startup_timeout_s}s",
        )

    async def _container_running(self) -> bool:
        if self._container_id is None:
            return False
        code, out = await self._docker(
            ["inspect", "-f", "{{.State.Running}}", self._container_id],
            check=False,
        )
        return code == 0 and out.strip() == "true"

    async def _container_logs(self) -> str:
        if self._container_id is None:
            return ""
        _, out = await self._docker(["logs", "--tail", "40", self._container_id], check=False)
        return out.strip()
