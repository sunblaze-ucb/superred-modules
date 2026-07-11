"""Per-instance host state: an isolated state directory + the env vars that let the
vendored compose / ``setup.sh`` mount the shared workspace.

Each DTAP instance gets an isolated host state directory
``${DTAP_STATE_ROOT or tempdir}/dtap/{iid}/`` with a ``workspace`` subdir, so
parallel instances never share mutable backend state.

For the filesystem domains (``os-filesystem`` / ``code`` / ``research``) the agent's
native bash tool, the env MCP server's file tools, and the OOB judge must agree on
the bytes on disk. That sharing is driven by :meth:`InstanceState.env_overrides`,
which exports the host workspace path (``DTAP_HOST_WORKSPACE``) and the in-container
mount path (``DTAP_WORKSPACE``, default :data:`SHARED_FS_CONTAINER_PATH`, override
with ``$DTAP_SHARED_FS_PATH``) to the vendored compose + setup scripts -- they do
the actual bind mount. No mount spec is produced here.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

# The container path the vendored compose/setup mount the shared workspace at.
SHARED_FS_CONTAINER_PATH: str = os.getenv("DTAP_SHARED_FS_PATH", "/workspace")


def sanitize_name(name: str) -> str:
    """Lowercase + collapse to ``[a-z0-9_-]`` for Docker volume/project names."""
    return re.sub(r"[^a-z0-9_-]", "_", name.lower())


def _state_base(state_root: str | os.PathLike[str] | None) -> Path:
    base = state_root or os.getenv("DTAP_STATE_ROOT") or os.path.join(tempfile.gettempdir(), "dtap")
    return Path(base)


@dataclass(frozen=True)
class InstanceState:
    """Host paths and Docker volume/mount names for one running instance."""

    iid: str
    state_dir: Path
    workspace_dir: Path

    def env_overrides(self) -> dict[str, str]:
        """Env vars exported to ``setup.sh`` / compose so they can mount the workspace.

        ``DTAP_INSTANCE_ID`` / ``DTAP_STATE_DIR`` / ``DTAP_HOST_WORKSPACE`` /
        ``DTAP_WORKSPACE`` let the vendored compose + setup scripts find the shared
        host directory and the container path it lands at.
        """
        return {
            "DTAP_INSTANCE_ID": self.iid,
            "DTAP_STATE_DIR": str(self.state_dir),
            "DTAP_HOST_WORKSPACE": str(self.workspace_dir),
            "DTAP_WORKSPACE": SHARED_FS_CONTAINER_PATH,
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
    "SHARED_FS_CONTAINER_PATH",
    "InstanceState",
    "make_instance_state",
    "sanitize_name",
]
