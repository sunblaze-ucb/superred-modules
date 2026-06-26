"""Thin async wrappers over ``docker compose`` (up / down / ps / health-wait).

Mirrors the upstream ``utils/task_executor.py`` compose handling: ``up -d`` with
the env's host-port variables exported, ``down --remove-orphans --volumes`` for
teardown, ``ps --format json`` polled until every container is running/healthy,
and ``sudo`` auto-detected when the daemon needs it. The ``-hub`` compose files
(referenced from ``env.yaml``) pull prebuilt images from Docker Hub, so ``up``
passes ``--no-build`` after a best-effort ``pull``.

EVERY subprocess invocation goes through :func:`_exec`, the single seam tests
monkeypatch to run the whole lifecycle without a Docker daemon.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

PULL_TIMEOUT = 1800
UP_TIMEOUT = 900
DOWN_TIMEOUT = 120
PS_TIMEOUT = 30

_sudo_cache: bool | None = None


class ComposeError(RuntimeError):
    """A ``docker compose`` invocation returned a non-zero exit code."""


async def _exec(
    cmd: list[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> tuple[int, str, str]:
    """Run *cmd* and return ``(returncode, stdout, stderr)``.

    THE single subprocess seam: every Docker/compose/script call below funnels
    through here, so a test can monkeypatch this one function to fake the daemon.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return (
        proc.returncode if proc.returncode is not None else -1,
        out.decode(errors="replace"),
        err.decode(errors="replace"),
    )


async def needs_sudo() -> bool:
    """Whether docker requires ``sudo`` here (cached; mirrors upstream detection)."""
    global _sudo_cache
    if _sudo_cache is not None:
        return _sudo_cache
    rc, _, _ = await _exec(["docker", "ps"], timeout=5)
    if rc == 0:
        _sudo_cache = False
        return False
    try:
        import grp

        if grp.getgrnam("docker").gr_gid in os.getgroups():
            _sudo_cache = False
            return False
    except (KeyError, OSError, ImportError):
        pass
    _sudo_cache = True
    return True


def _compose_cmd(
    project: str,
    compose_file: str | os.PathLike[str],
    args: list[str],
    *,
    sudo: bool,
    ports: dict[str, int] | None = None,
) -> list[str]:
    """Build a ``docker compose`` argv, optionally wrapped in ``sudo env VAR=...``."""
    base = ["docker", "compose", "-p", project, "-f", str(compose_file)]
    if sudo:
        env_args = [f"{k}={v}" for k, v in (ports or {}).items()]
        return ["sudo", "env", *env_args, *base, *args]
    return [*base, *args]


def _run_env(ports: dict[str, int] | None) -> dict[str, str]:
    env = dict(os.environ)
    for var, port in (ports or {}).items():
        env[var] = str(port)
    return env


async def compose_up(
    project: str,
    compose_file: str | os.PathLike[str],
    *,
    ports: dict[str, int] | None = None,
    sudo: bool | None = None,
    pull: bool = True,
) -> None:
    """``docker compose -p <project> -f <file> up -d --no-build`` with *ports* exported."""
    if sudo is None:
        sudo = await needs_sudo()
    cwd = Path(compose_file).parent
    run_env = None if sudo else _run_env(ports)

    if pull:
        # Best-effort image pull; ignore failures (images may already be local).
        await _exec(
            _compose_cmd(project, compose_file, ["pull"], sudo=sudo, ports=ports),
            cwd=cwd,
            env=run_env,
            timeout=PULL_TIMEOUT,
        )

    rc, _, err = await _exec(
        _compose_cmd(
            project, compose_file, ["up", "-d", "--no-build"], sudo=sudo, ports=ports
        ),
        cwd=cwd,
        env=run_env,
        timeout=UP_TIMEOUT,
    )
    if rc != 0:
        raise ComposeError(f"compose up failed for project {project!r}: {err.strip()}")


async def compose_down(
    project: str,
    compose_file: str | os.PathLike[str],
    *,
    sudo: bool | None = None,
) -> None:
    """``docker compose -p <project> -f <file> down --remove-orphans --volumes``."""
    if sudo is None:
        sudo = await needs_sudo()
    await _exec(
        _compose_cmd(
            project,
            compose_file,
            ["down", "--remove-orphans", "--volumes"],
            sudo=sudo,
        ),
        cwd=Path(compose_file).parent,
        timeout=DOWN_TIMEOUT,
    )


def _parse_ps_json(output: str) -> list[dict[str, Any]]:
    """Parse ``compose ps --format json`` (a JSON array OR newline-delimited objects)."""
    text = output.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


async def compose_ps(
    project: str,
    compose_file: str | os.PathLike[str],
    *,
    sudo: bool | None = None,
) -> list[dict[str, Any]]:
    """Return the parsed ``compose ps --format json`` rows (empty on failure)."""
    if sudo is None:
        sudo = await needs_sudo()
    rc, out, _ = await _exec(
        _compose_cmd(project, compose_file, ["ps", "--format", "json"], sudo=sudo),
        cwd=Path(compose_file).parent,
        timeout=PS_TIMEOUT,
    )
    if rc != 0:
        return []
    return _parse_ps_json(out)


def _row_ready(row: dict[str, Any]) -> bool:
    """A container is ready if running and (no healthcheck OR healthy)."""
    state = str(row.get("State", "")).lower()
    health = str(row.get("Health", "")).lower()
    if state != "running":
        return False
    return health in ("", "healthy")


async def wait_healthy(
    project: str,
    compose_file: str | os.PathLike[str],
    *,
    sudo: bool | None = None,
    timeout: float = 120,
    interval: float = 2.0,
) -> bool:
    """Poll ``compose ps`` until all containers are healthy or *timeout* elapses."""
    if sudo is None:
        sudo = await needs_sudo()
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        rows = await compose_ps(project, compose_file, sudo=sudo)
        if rows and all(_row_ready(r) for r in rows):
            return True
        await asyncio.sleep(interval)
    return False


def reset_sudo_cache() -> None:
    """Forget the cached sudo detection (used by tests)."""
    global _sudo_cache
    _sudo_cache = None


__all__ = [
    "ComposeError",
    "needs_sudo",
    "compose_up",
    "compose_down",
    "compose_ps",
    "wait_healthy",
    "reset_sudo_cache",
]
