"""DTAP Docker + MCP environment lifecycle.

The :class:`~dtap_scaffold.docker.lifecycle.DockerEnvStack` is the concrete
``EnvStack`` the agent-agnostic base spins up per task. It is backed by the
sibling modules in this package:

- :mod:`env_registry` -- parse ``mcp.yaml`` / ``env.yaml`` / ``injection_mcp.yaml``.
- :mod:`ports`        -- lease distinct free host ports (bind test + file lock).
- :mod:`compose`      -- ``docker compose up/down/ps`` + health-wait (one seam).
- :mod:`state`        -- per-instance state dir, named volumes, shared FS mount.
- :mod:`reset`        -- between-run backend reset (endpoints then scripts).
- :mod:`lifecycle`    -- the ``DockerEnvStack`` orchestrating all of the above.
"""

from __future__ import annotations

from dtap_scaffold.docker.env_registry import (
    EnvRegistry,
    EnvRegistryError,
    load,
    mcp_port_key,
    resolve_config_dir,
)
from dtap_scaffold.docker.lifecycle import DockerEnvStack
from dtap_scaffold.docker.ports import PortLeaser
from dtap_scaffold.docker.state import InstanceState, make_instance_state

__all__ = [
    "DockerEnvStack",
    "EnvRegistry",
    "EnvRegistryError",
    "load",
    "resolve_config_dir",
    "mcp_port_key",
    "PortLeaser",
    "InstanceState",
    "make_instance_state",
]
