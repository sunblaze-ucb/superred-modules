"""Static DTAP environment registry: resolve+parse the three vendored config YAMLs.

The upstream ``decodingtrust-agent-sdk`` ships three YAMLs under ``dt_arena/config``:

- ``mcp.yaml``           -- the env MCP servers (name -> ``environment`` -> command/env).
- ``env.yaml``           -- the Docker environments (compose file, ports, reset, limits).
- ``injection_mcp.yaml`` -- the ``<server>-injection`` red-teaming MCP servers.

This module loads those three files and answers the static questions the Docker
lifecycle needs: which Docker environment(s) back an active MCP server, where the
compose file is, which host-port variables that environment maps, how to reset
it, and which injection servers a task's ``env_injection_config`` requires. It
also enforces the text-only domain allowlist (rejecting the vision/GUI
``browser`` / ``macos`` / ``windows`` environments) via
:mod:`dtap_scaffold.text_domains`.

Config location resolution (mirrors the upstream ``utils/config.py``): an
explicit ``config_dir`` argument wins, then ``$DTAP_ROOT/dt_arena/config``, then
the installed SDK via ``importlib.resources.files("dt_arena")/"config"``, then
``$DT_ROOT/dt_arena/config`` (the clone, used by the offline tests). Nothing here
runs Docker or the network -- it is pure parsing.
"""

from __future__ import annotations

import importlib.resources
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from dtap_scaffold.text_domains import require_text_only_domain

# Vision/GUI Docker environments and the DTAP domain each belongs to. These are
# the ONLY environments this text-only port rejects; every other environment is
# text-based (see dtap_scaffold.text_domains). ``browser`` is backed by the
# ``ecommerce`` + ``custom-website`` envs; ``macos`` / ``windows`` by their VMs.
_GUI_ENVIRONMENTS: dict[str, str] = {
    "ecommerce": "browser",
    "custom-website": "browser",
    "macos": "macos",
    "windows": "windows",
}

_DEFAULT_HEALTH_TIMEOUT = 120


class EnvRegistryError(RuntimeError):
    """Raised when the registry cannot resolve or parse the DTAP config YAMLs."""


def resolve_config_dir(config_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the ``dt_arena/config`` directory holding the three YAMLs.

    Order: explicit *config_dir* > ``$DTAP_ROOT/dt_arena/config`` > installed SDK
    (``importlib.resources``) > ``$DT_ROOT/dt_arena/config`` (the test clone).
    """
    if config_dir is not None:
        p = Path(config_dir)
        if not p.is_dir():
            raise EnvRegistryError(f"config_dir does not exist: {p}")
        return p

    env_root = os.environ.get("DTAP_ROOT")
    if env_root:
        cand = Path(env_root) / "dt_arena" / "config"
        if cand.is_dir():
            return cand

    try:
        ref = importlib.resources.files("dt_arena") / "config"
        with importlib.resources.as_file(ref) as p:
            if Path(p).is_dir():
                return Path(p)
    except (ModuleNotFoundError, FileNotFoundError, TypeError, AttributeError):
        pass

    dt_root = os.environ.get("DT_ROOT")
    if dt_root:
        cand = Path(dt_root) / "dt_arena" / "config"
        if cand.is_dir():
            return cand

    raise EnvRegistryError(
        "could not locate dt_arena/config; set DTAP_ROOT or DT_ROOT to the SDK/clone "
        "root, install decodingtrust-agent-sdk, or pass config_dir explicitly."
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EnvRegistryError(f"missing config file: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


class EnvRegistry:
    """Parsed, read-only view over ``mcp.yaml`` / ``env.yaml`` / ``injection_mcp.yaml``."""

    def __init__(self, config_dir: str | os.PathLike[str] | None = None) -> None:
        self.config_dir = resolve_config_dir(config_dir)
        self._mcp = _load_yaml(self.config_dir / "mcp.yaml")
        self._env = _load_yaml(self.config_dir / "env.yaml")
        self._inj = _load_yaml(self.config_dir / "injection_mcp.yaml")

        # Case-insensitive name maps (mirrors upstream build_server_name_map).
        self._mcp_by_name = {
            s["name"].lower(): s
            for s in (self._mcp.get("servers") or [])
            if s.get("name")
        }
        self._inj_by_name = {
            s["name"].lower(): s
            for s in (self._inj.get("servers") or [])
            if s.get("name")
        }
        self._environments: dict[str, Any] = self._env.get("environments") or {}
        self.default_max_instances = self._env.get("default_max_instances")

    # ----- SDK root + globals ----------------------------------------------

    @property
    def sdk_root(self) -> Path:
        """The SDK/clone root (``dt_arena/config`` -> parents[1])."""
        return self.config_dir.parent.parent

    @property
    def env_config(self) -> dict[str, Any]:
        """The raw parsed ``env.yaml`` (shape :func:`reset.reset_environment` expects)."""
        return self._env

    def mcp_base_dir(self) -> Path:
        """Directory holding the env MCP server trees (``global.base_dir``)."""
        base = (self._mcp.get("global") or {}).get("base_dir", "../mcp_server")
        return (self.config_dir / base).resolve()

    def injection_base_dir(self) -> Path:
        """Directory holding the injection MCP server trees (``global.base_dir``)."""
        base = (self._inj.get("global") or {}).get(
            "base_dir", "../injection_mcp_server"
        )
        return (self.config_dir / base).resolve()

    # ----- mcp.yaml ---------------------------------------------------------

    def mcp_server(self, name: str) -> dict[str, Any] | None:
        """The mcp.yaml server entry for *name* (case-insensitive)."""
        return self._mcp_by_name.get(name.lower())

    def server_environments(self, name: str) -> list[str]:
        """Docker environment(s) backing MCP server *name* (str or list in YAML)."""
        cfg = self.mcp_server(name)
        if cfg is None:
            raise EnvRegistryError(f"unknown MCP server: {name!r}")
        env = cfg.get("environment")
        if env is None:
            return []
        return [env] if isinstance(env, str) else [str(e) for e in env]

    def active_environments(
        self, active_servers: list[str] | tuple[str, ...]
    ) -> list[str]:
        """Deduplicated, order-preserving environments for *active_servers*."""
        seen: list[str] = []
        for server in active_servers:
            for env in self.server_environments(server):
                if env not in seen:
                    seen.append(env)
        return seen

    # ----- env.yaml ---------------------------------------------------------

    def environment(self, env_name: str) -> dict[str, Any]:
        """The env.yaml entry for *env_name*."""
        if env_name not in self._environments:
            raise EnvRegistryError(f"unknown Docker environment: {env_name!r}")
        return self._environments[env_name]

    def compose_file(self, env_name: str) -> Path:
        """Absolute path to *env_name*'s docker-compose file (relative to SDK root)."""
        rel = self.environment(env_name).get("docker_compose")
        if not rel:
            raise EnvRegistryError(f"no docker_compose for environment {env_name!r}")
        return (self.sdk_root / rel).resolve()

    def env_ports(self, env_name: str) -> dict[str, dict[str, Any]]:
        """Host-port variable map ``{VAR: {default, container_port}}`` for *env_name*."""
        return dict(self.environment(env_name).get("ports") or {})

    def reset_endpoints(self, env_name: str) -> dict[str, Any]:
        return dict(self.environment(env_name).get("reset_endpoints") or {})

    def reset_scripts(self, env_name: str) -> dict[str, Any]:
        return dict(self.environment(env_name).get("reset_scripts") or {})

    def max_instances(self, env_name: str) -> int | None:
        return self.environment(env_name).get(
            "max_instances", self.default_max_instances
        )

    def disable_reuse(self, env_name: str) -> bool:
        return bool(self.environment(env_name).get("disable_reuse", False))

    def health_timeout(self, env_name: str) -> int:
        return int(
            self.environment(env_name).get("health_timeout", _DEFAULT_HEALTH_TIMEOUT)
        )

    def get_compose_files(
        self, active_servers: list[str] | tuple[str, ...]
    ) -> set[Path]:
        """Set of compose files for the environments backing *active_servers*."""
        out: set[Path] = set()
        for env in self.active_environments(active_servers):
            if env in self._environments and self._environments[env].get(
                "docker_compose"
            ):
                out.add(self.compose_file(env))
        return out

    # ----- injection_mcp.yaml ----------------------------------------------

    def injection_server(self, name: str) -> dict[str, Any] | None:
        """The injection_mcp.yaml entry for *name* (case-insensitive)."""
        return self._inj_by_name.get(name.lower())

    def injection_target_environment(self, name: str) -> str:
        """Docker environment an injection server writes to.

        Prefers the explicit ``target_environment`` field; otherwise strips the
        ``-injection`` suffix (e.g. ``os-filesystem-injection`` -> ``os-filesystem``).
        """
        cfg = self.injection_server(name)
        if cfg and cfg.get("target_environment"):
            return str(cfg["target_environment"])
        base = name.lower()
        suffix = "-injection"
        return base[: -len(suffix)] if base.endswith(suffix) else base

    def required_injection_servers(
        self, env_injection_config: dict[str, Any] | None
    ) -> dict[str, dict[str, Any]]:
        """Map each requested injection-server name to its injection_mcp.yaml entry.

        *env_injection_config* is the task's ``RedTeamingAgent.env_injection_config``
        (keys are injection-server names). Unknown names are dropped.
        """
        out: dict[str, dict[str, Any]] = {}
        for name in env_injection_config or {}:
            cfg = self.injection_server(name)
            if cfg is not None:
                out[name] = cfg
        return out

    # ----- URLs + text-only validation -------------------------------------

    @staticmethod
    def server_url(server: str, port: int, host: str = "127.0.0.1") -> str:
        """The MCP URL the proxy/agent connects to (streamable-http ``/mcp`` path)."""
        return f"http://{host}:{port}/mcp"

    def domain_for_server(self, name: str) -> str | None:
        """The excluded DTAP domain a server maps to, or ``None`` if text-only.

        Only vision/GUI environments yield a domain name here (``browser`` /
        ``macos`` / ``windows``); text-only servers return ``None``.
        """
        for env in self.server_environments(name):
            if env in _GUI_ENVIRONMENTS:
                return _GUI_ENVIRONMENTS[env]
        return None

    def require_text_only(self, active_servers: list[str] | tuple[str, ...]) -> None:
        """Raise ``ValueError`` if any active server is a vision/GUI domain.

        Uses :func:`dtap_scaffold.text_domains.require_text_only_domain` so the
        rejection message is the canonical one.
        """
        for server in active_servers:
            domain = self.domain_for_server(server)
            if domain is not None:
                require_text_only_domain(domain)  # raises for browser/macos/windows


def mcp_port_key(server_cfg: dict[str, Any], prefix: str = "mcp") -> str:
    """The env-var holding a server's own listen port (mirrors upstream selection).

    Default ``PORT``; if an env key contains both ``PORT`` and the *prefix*
    (e.g. ``TELECOM_MCP_PORT`` for the ``mcp`` prefix), that key wins.
    """
    env = server_cfg.get("env") or {}
    for key in env:
        upper = key.upper()
        if "PORT" in upper and prefix.upper() in upper:
            return key
    return "PORT"


@lru_cache(maxsize=None)
def _cached_registry(config_dir: str | None) -> EnvRegistry:
    return EnvRegistry(config_dir)


def load(config_dir: str | os.PathLike[str] | None = None) -> EnvRegistry:
    """Return a cached :class:`EnvRegistry` for *config_dir* (resolved if ``None``)."""
    resolved = str(resolve_config_dir(config_dir))
    return _cached_registry(resolved)


__all__ = [
    "EnvRegistry",
    "EnvRegistryError",
    "resolve_config_dir",
    "mcp_port_key",
    "load",
]
