"""``DockerEnvStack``: the per-instance Docker + MCP environment lifecycle.

This is the concrete :class:`~dtap_scaffold.protocols.EnvStack` the agent-agnostic
base creates. One stack owns one task's whole runtime environment:

``up()``    -- mint an instance id, lease host ports, ``docker compose up`` the env
              containers for the task's active servers (each under project
              ``dtap_{iid}_{env}``), run the task's ``setup.sh`` to seed data,
              then start the per-server env MCP processes and the injection MCP
              processes, and return an :class:`~dtap_scaffold.types.EnvHandle`.
``reset()`` -- reset each env's backend (endpoints/scripts) and re-run ``setup.sh``;
              containers + leases are kept.
``down()``  -- stop the MCP processes, ``docker compose down --volumes``, release
              the port leases.

Every Docker/subprocess/HTTP effect is delegated to the sibling modules
(:mod:`compose`, :mod:`reset`, :mod:`ports`, :mod:`state`, :mod:`env_registry`)
or to the two module-level seams here (:func:`_spawn_process`,
:func:`_wait_for_ready`); tests monkeypatch those to drive the full ordering
without a Docker daemon.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dtap_scaffold.docker import compose, env_registry, reset
from dtap_scaffold.docker import ports as ports_mod
from dtap_scaffold.docker import state as state_mod
from dtap_scaffold.types import EnvHandle

SETUP_TIMEOUT = 600
_READY_TIMEOUT = 150.0


# --------------------------- module-level seams ---------------------------


def _spawn_process(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    log_path: str | None = None,
) -> "subprocess.Popen[bytes]":
    """Start a long-lived MCP server process (the process-spawn seam).

    When *log_path* is given, the server's stdout+stderr are written there (so a
    crash-on-start, e.g. a missing dependency, is diagnosable instead of vanishing
    into DEVNULL and turning into a readiness timeout).
    """
    out: Any = subprocess.DEVNULL
    if log_path:
        out = open(log_path, "wb")  # noqa: SIM115 - handle closed when the process exits
    return subprocess.Popen(  # noqa: S603 (cmd built from vendored config)
        cmd,
        cwd=cwd,
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT if log_path else subprocess.DEVNULL,
        start_new_session=True,
    )


def _terminate_process(proc: Any) -> None:
    """Best-effort terminate of a spawned process (never raises)."""
    try:
        proc.terminate()
    except Exception:  # noqa: BLE001 - teardown must not raise
        pass


def _is_listening(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


async def _wait_for_ready(
    server_urls: dict[str, str],
    *,
    timeout: float = _READY_TIMEOUT,
    interval: float = 0.5,
) -> None:
    """Block until every URL accepts a TCP connection or *timeout* elapses (the readiness seam)."""
    if not server_urls:
        return
    pending = dict(server_urls)
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while pending and loop.time() < deadline:
        for name, url in list(pending.items()):
            if _is_listening(url):
                pending.pop(name, None)
        if pending:
            await asyncio.sleep(interval)
    if pending:
        raise RuntimeError(
            f"MCP servers failed to become ready: {', '.join(sorted(pending))}"
        )


# --------------------------- templating helpers ---------------------------


def _server_env(
    cfg: dict[str, Any],
    port_key: str,
    listen: int,
    container_ports: dict[str, int],
    extra: dict[str, str],
) -> dict[str, str]:
    """Build an MCP server's process env: own listen port + rendered ``${VAR}`` ports."""
    env = dict(os.environ)
    render_values: dict[str, int] = {**container_ports, port_key: listen}
    env[port_key] = str(listen)
    for key, value in (cfg.get("env") or {}).items():
        if key == port_key:
            continue
        env[key] = reset.render_template(str(value), render_values)
    env.update(extra)
    return env


def _expand_command(cfg: dict[str, Any], env: dict[str, str]) -> list[str]:
    """Expand ``$VAR`` / ``${VAR}`` in a server's command (upstream MCPServerManager parity)."""
    cmd: list[str] = []
    for part in cfg.get("command") or []:
        expanded = str(part)
        for key, value in env.items():
            expanded = expanded.replace(f"${{{key}}}", str(value)).replace(
                f"${key}", str(value)
            )
        cmd.append(expanded)
    return cmd


# --------------------------- the env stack --------------------------------


class DockerEnvStack:
    """Per-instance Docker + MCP environment (implements ``EnvStack``)."""

    def __init__(
        self,
        active_servers: tuple[str, ...] | list[str],
        injection_config: dict[str, Any] | None,
        task_dir: str | os.PathLike[str] | None,
        state_root: str | os.PathLike[str] | None = None,
        *,
        config_dir: str | os.PathLike[str] | None = None,
        host: str = "127.0.0.1",
    ) -> None:
        self._active_servers = tuple(active_servers)
        self._injection_config = dict(injection_config or {})
        self._task_dir = Path(task_dir) if task_dir else None
        self._state_root = state_root
        self._host = host
        # The env registry is loaded lazily (in up()/require_text_only via the
        # _registry property), so a target can be CONSTRUCTED and inspected
        # (get_controllables / security_domain / get_observables) without the SDK
        # or dt_arena/config present; it is only needed once Docker actually runs.
        self._config_dir = config_dir
        self._registry_cache: Any = None

        # Runtime state (populated by up()).
        self._iid = ""
        self._state: state_mod.InstanceState | None = None
        self._leaser = ports_mod.PortLeaser()
        self._sudo: bool | None = None
        self._container_ports: dict[str, int] = {}  # env.yaml VAR -> leased host port
        self._projects: dict[str, str] = {}  # env name -> compose project name
        self._mcp_procs: dict[str, Any] = {}  # server name -> process
        self._inj_procs: dict[str, Any] = {}  # injection server name -> process
        self._server_urls: dict[str, str] = {}
        self._inj_urls: dict[str, str] = {}
        self._server_logs: dict[str, str] = {}  # server name -> stdout/stderr log path
        self._handle: EnvHandle | None = None
        self._up_done = False

    @property
    def _registry(self) -> Any:
        """The env registry, loaded lazily on first use (needs dt_arena/config)."""
        if self._registry_cache is None:
            self._registry_cache = env_registry.load(self._config_dir)
        return self._registry_cache

    # ----- EnvStack protocol -----------------------------------------------

    async def up(self) -> EnvHandle:
        if self._up_done and self._handle is not None:
            return self._handle

        # Reject any vision/GUI server before touching Docker.
        self._registry.require_text_only(self._active_servers)

        self._iid = uuid.uuid4().hex[:8]
        self._state = state_mod.make_instance_state(
            self._iid, state_root=self._state_root
        )
        self._sudo = await compose.needs_sudo()

        await self._bring_up_environments()
        await self._run_setup()
        await self._start_mcp_servers()
        await self._start_injection_servers()
        try:
            await _wait_for_ready({**self._server_urls, **self._inj_urls})
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}\n{self._server_log_tails()}") from exc

        self._handle = EnvHandle(
            server_urls=dict(self._server_urls),
            injection_server_urls=dict(self._inj_urls),
            ports=dict(self._container_ports),
            network=None,
        )
        self._up_done = True
        return self._handle

    async def reset(self) -> None:
        env_config = self._registry.env_config
        for env, project in self._projects.items():
            try:
                compose_file = self._registry.compose_file(env)
            except env_registry.EnvRegistryError:
                compose_file = None
            await reset.reset_environment(
                env,
                self._container_ports,
                env_config,
                project_name=project,
                compose_file=compose_file,
                sudo=self._sudo,
            )
        await self._run_setup()

    async def down(self) -> None:
        for proc in (*self._mcp_procs.values(), *self._inj_procs.values()):
            _terminate_process(proc)
        self._mcp_procs.clear()
        self._inj_procs.clear()

        for env, project in list(self._projects.items()):
            try:
                await compose.compose_down(
                    project, self._registry.compose_file(env), sudo=self._sudo
                )
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
        self._projects.clear()
        self._leaser.release_all()
        self._server_urls.clear()
        self._inj_urls.clear()
        self._up_done = False

    # ----- up() internals --------------------------------------------------

    async def _bring_up_environments(self) -> None:
        for env in self._registry.active_environments(self._active_servers):
            # Lease every host-port variable this environment exposes.
            env_ports: dict[str, int] = {}
            for var in self._registry.env_ports(env):
                port = self._leaser.lease(var)
                self._container_ports[var] = port
                env_ports[var] = port

            try:
                compose_file = self._registry.compose_file(env)
            except env_registry.EnvRegistryError:
                continue  # no compose file -> nothing to bring up for this env

            project = f"dtap_{self._iid}_{state_mod.sanitize_name(env)}"
            self._projects[env] = project
            await compose.compose_up(
                project, compose_file, ports=env_ports, sudo=self._sudo
            )
            await compose.wait_healthy(
                project,
                compose_file,
                sudo=self._sudo,
                timeout=self._registry.health_timeout(env),
            )

    async def _run_setup(self) -> None:
        if self._task_dir is None:
            return
        setup = self._task_dir / "setup.sh"
        if not setup.is_file():
            return
        run_env = dict(os.environ)
        run_env.update({k: str(v) for k, v in self._container_ports.items()})
        if self._state is not None:
            run_env.update(self._state.env_overrides())
        rc, _, err = await compose._exec(
            ["bash", str(setup)],
            cwd=str(self._task_dir),
            env=run_env,
            timeout=SETUP_TIMEOUT,
        )
        if rc != 0:
            raise RuntimeError(f"setup.sh failed ({self._task_dir}): {err.strip()}")

    async def _start_mcp_servers(self) -> None:
        for server in self._active_servers:
            cfg = self._registry.mcp_server(server)
            if cfg is None:
                raise env_registry.EnvRegistryError(f"unknown MCP server: {server!r}")
            url = self._launch(
                server, cfg, prefix="mcp", base_dir=self._registry.mcp_base_dir()
            )
            self._server_urls[server] = url

    async def _start_injection_servers(self) -> None:
        required = self._registry.required_injection_servers(self._injection_config)
        for name, cfg in required.items():
            url = self._launch(
                name,
                cfg,
                prefix="injection",
                base_dir=self._registry.injection_base_dir(),
            )
            self._inj_urls[name] = url

    def _project_name_overrides(self) -> dict[str, str]:
        """Export ``<ENV>_PROJECT_NAME`` for every active env (upstream pool convention).

        DTAP servers that ``docker exec`` into their env container (terminal,
        research, os-filesystem, ...) resolve the container name from
        ``f"{env.upper().replace('-', '_')}_PROJECT_NAME"`` (mirrors
        ``utils.compose_utils.get_project_name``) -> ``{project}-{env}-env-1``.
        Upstream's environment pool sets these in the parent process env BEFORE the
        MCP manager starts each server; this stack plays the pool's role, so it
        exports the same vars, each set to its env's per-instance compose project.
        Without them the terminal/research servers raise
        ``"<ENV>_PROJECT_NAME is not set"`` on start.
        """
        return {
            f"{env.upper().replace('-', '_')}_PROJECT_NAME": project
            for env, project in self._projects.items()
        }

    def _launch(
        self, name: str, cfg: dict[str, Any], *, prefix: str, base_dir: Path
    ) -> str:
        """Lease a listen port, spawn the server process, return its ``/mcp`` URL."""
        port_key = env_registry.mcp_port_key(cfg, prefix)
        listen = self._leaser.lease(f"{prefix}.{name.lower()}")
        extra = dict(self._state.env_overrides()) if self._state is not None else {}
        extra.update(
            self._project_name_overrides()
        )  # <ENV>_PROJECT_NAME for exec-based servers
        env = _server_env(cfg, port_key, listen, self._container_ports, extra)
        cmd = _expand_command(cfg, env)
        cwd = base_dir / Path(cfg["path"]).parent
        log_path = str(
            self._logs_dir() / f"{prefix}_{state_mod.sanitize_name(name)}.log"
        )
        self._server_logs[name] = log_path
        proc = _spawn_process(cmd, cwd=str(cwd), env=env, log_path=log_path)
        (self._inj_procs if prefix == "injection" else self._mcp_procs)[name] = proc
        return self._registry.server_url(name, listen, host=self._host)

    def _logs_dir(self) -> Path:
        base = (
            Path(self._state_root) if self._state_root else Path(tempfile.gettempdir())
        )
        directory = base / f"dtap_logs_{self._iid}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _server_log_tails(self, lines: int = 25) -> str:
        """Last lines of each spawned server's log (for actionable readiness errors)."""
        out: list[str] = []
        for name, path in self._server_logs.items():
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    tail = "".join(fh.readlines()[-lines:]).strip()
            except OSError:
                tail = "(no log)"
            out.append(f"--- {name} ({path}) ---\n{tail}")
        return "\n".join(out) if out else "(no server logs captured)"


__all__ = ["DockerEnvStack"]
