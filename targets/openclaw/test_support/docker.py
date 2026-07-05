"""Docker daemon and managed-target helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from openclaw_target.docker_runtime import DEFAULT_DOCKER_IMAGE
from openclaw_target.target import OpenClawTarget


def docker_daemon_ready() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(
        ["docker", "info"],
        capture_output=True,
        check=False,
    ).returncode == 0


def docker_image() -> str:
    return os.environ.get("OPENCLAW_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)


def docker_target(**kwargs: Any) -> OpenClawTarget:
    """Build an :class:`OpenClawTarget` on the containerised managed runtime."""
    return OpenClawTarget(
        managed=True,
        managed_runtime="docker",
        managed_kwargs={"image": docker_image()},
        **kwargs,
    )
