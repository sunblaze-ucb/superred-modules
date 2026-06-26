"""Per-instance state: a host state dir, named state volumes, and the shared FS mount.

Each DTAP instance gets an isolated host state directory
``${DTAP_STATE_ROOT or tempdir}/dtap/{iid}/`` and a per-environment named volume
``dtap_{iid}_{env}_state`` so parallel instances never share mutable backend
state.

The three filesystem domains -- ``os-filesystem``, ``code``, ``research`` -- have
a twist: the agent's NATIVE bash tool, the env MCP server's file tools, and the
out-of-band judge all touch the SAME files. They must therefore agree on the
bytes on disk. :meth:`InstanceState.shared_fs_mount` produces a host-bind mount
spec (host ``workspace`` dir -> a fixed container path) that the env container,
the agent container, and the judge all mount at the identical path. The container
path is :data:`SHARED_FS_CONTAINER_PATH` (override with ``$DTAP_SHARED_FS_PATH``).
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

# DTAP domains whose backend IS a filesystem the agent also touches via bash.
FS_SHARED_DOMAINS: frozenset[str] = frozenset({"os-filesystem", "code", "research"})

# The container path every collaborator mounts the shared workspace at.
SHARED_FS_CONTAINER_PATH: str = os.getenv("DTAP_SHARED_FS_PATH", "/workspace")


def is_fs_shared_domain(domain: str | None) -> bool:
    """Whether *domain* needs the shared host-bind workspace mount."""
    return domain in FS_SHARED_DOMAINS


def sanitize_name(name: str) -> str:
    """Lowercase + collapse to ``[a-z0-9_-]`` for Docker volume/project names."""
    return re.sub(r"[^a-z0-9_-]", "_", name.lower())


# Backwards-friendly internal alias.
_sanitize = sanitize_name


def _state_base(state_root: str | os.PathLike[str] | None) -> Path:
    base = (
        state_root
        or os.getenv("DTAP_STATE_ROOT")
        or os.path.join(tempfile.gettempdir(), "dtap")
    )
    return Path(base)


@dataclass(frozen=True)
class InstanceState:
    """Host paths and Docker volume/mount names for one running instance."""

    iid: str
    state_dir: Path
    workspace_dir: Path

    def volume_name(self, env: str) -> str:
        """Named Docker volume for *env*'s mutable state: ``dtap_{iid}_{env}_state``."""
        return f"dtap_{self.iid}_{_sanitize(env)}_state"

    def shared_fs_mount(
        self, *, container_path: str = SHARED_FS_CONTAINER_PATH
    ) -> dict[str, str]:
        """Host-bind mount spec sharing the workspace at the identical container path."""
        return {
            "type": "bind",
            "source": str(self.workspace_dir),
            "target": container_path,
        }

    def env_overrides(
        self, *, container_path: str = SHARED_FS_CONTAINER_PATH
    ) -> dict[str, str]:
        """Env vars exported to ``setup.sh`` / compose so they can mount the workspace.

        ``DTAP_INSTANCE_ID`` / ``DTAP_STATE_DIR`` / ``DTAP_HOST_WORKSPACE`` /
        ``DTAP_WORKSPACE`` let the vendored compose + setup scripts find the shared
        host directory and the container path it lands at.
        """
        return {
            "DTAP_INSTANCE_ID": self.iid,
            "DTAP_STATE_DIR": str(self.state_dir),
            "DTAP_HOST_WORKSPACE": str(self.workspace_dir),
            "DTAP_WORKSPACE": container_path,
        }


def make_instance_state(
    iid: str, *, state_root: str | os.PathLike[str] | None = None
) -> InstanceState:
    """Create (mkdir) and return the :class:`InstanceState` for instance *iid*."""
    state_dir = (_state_base(state_root) / iid).resolve()
    workspace_dir = state_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return InstanceState(iid=iid, state_dir=state_dir, workspace_dir=workspace_dir)


__all__ = [
    "FS_SHARED_DOMAINS",
    "SHARED_FS_CONTAINER_PATH",
    "InstanceState",
    "is_fs_shared_domain",
    "make_instance_state",
    "sanitize_name",
]
