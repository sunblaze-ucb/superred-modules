"""DTAP Docker + MCP environment lifecycle.

The :class:`~dtap_scaffold.docker.lifecycle.DockerEnvStack` is the concrete
``EnvStack`` the agent-agnostic base spins up per task. It is backed by the
sibling modules in this package:

- :mod:`env_registry` -- parse ``mcp.yaml`` / ``env.yaml`` / ``injection_mcp.yaml``.
- :mod:`ports`        -- lease distinct free host ports (bind test + file lock).
- :mod:`compose`      -- ``docker compose up/down/ps`` + health-wait (one seam).
- :mod:`state`        -- per-instance host state dir + the workspace env exports.
- :mod:`reset`        -- between-run backend reset (endpoints then scripts).
- :mod:`lifecycle`    -- the ``DockerEnvStack`` orchestrating all of the above.

Callers import the concrete class from its submodule (e.g.
``from dtap_scaffold.docker.lifecycle import DockerEnvStack``); this package is a
namespace, not a re-export facade.
"""
