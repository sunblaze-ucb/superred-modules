"""Reset an environment's backend data between runs (endpoints, then scripts).

Faithful to the upstream ``utils/reset_helpers.py``: prefer the env's HTTP
``reset_endpoints`` (``${VAR}`` placeholders resolved from the leased ports),
falling back to ``reset_scripts`` run via ``docker compose exec`` when endpoints
are absent or fail. The two OS seams are :func:`_http_post` (HTTP) and
:func:`dtap_scaffold.docker.compose._exec` (the compose-exec subprocess); tests
monkeypatch both.
"""

from __future__ import annotations

import asyncio
import shlex
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dtap_scaffold.docker import compose


class ResetScriptError(RuntimeError):
    """A container reset script timed out or returned a non-zero exit code."""


def render_template(template: str, values: dict[str, int]) -> str:
    """Replace ``${VAR}`` placeholders in *template* with *values* (upstream parity)."""
    result = str(template)
    for var_name, value in values.items():
        result = result.replace(f"${{{var_name}}}", str(value))
    return result


def _http_post(url: str, *, method: str = "POST", timeout: float = 30) -> int:
    """Issue an empty-body request to *url*; return the HTTP status. The HTTP seam."""
    req = urllib.request.Request(url, method=method, data=b"")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310 (localhost)
        return int(response.status)


async def reset_via_endpoints(
    env_name: str,
    ports: dict[str, int],
    env_config: dict[str, Any],
    *,
    timeout: float = 30,
    max_retries: int = 10,
    retry_delay: float = 2.0,
) -> None:
    """Reset *env_name* by calling its HTTP reset endpoints (with retries)."""
    env_def = (env_config.get("environments") or {}).get(env_name, {})
    reset_endpoints = env_def.get("reset_endpoints") or {}
    if not reset_endpoints:
        raise RuntimeError(f"no reset endpoints configured for {env_name}")

    for endpoint_name, endpoint_config in reset_endpoints.items():
        method = str(endpoint_config.get("method", "POST")).upper()
        url = render_template(endpoint_config.get("url", ""), ports)
        if "${" in url:  # unresolved placeholder -> can't call this endpoint
            continue

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                status = await asyncio.to_thread(_http_post, url, method=method, timeout=timeout)
                if 200 <= status < 300:
                    break
                last_error = RuntimeError(
                    f"reset endpoint {endpoint_name!r} returned HTTP {status}"
                )
            except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
                last_error = RuntimeError(f"reset endpoint {endpoint_name!r} failed: {exc}")
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)
            elif last_error is not None:
                raise last_error


async def reset_via_scripts(
    env_name: str,
    project_name: str,
    compose_file: str | Path,
    env_config: dict[str, Any],
    *,
    sudo: bool | None = None,
    timeout: float = 30,
) -> None:
    """Reset *env_name* by running its ``reset_scripts`` via ``docker compose exec``."""
    env_def = (env_config.get("environments") or {}).get(env_name, {})
    reset_scripts = env_def.get("reset_scripts") or {}
    if not reset_scripts:
        raise RuntimeError(f"no reset scripts configured for {env_name}")
    if sudo is None:
        sudo = await compose.needs_sudo()

    for service, script_path in reset_scripts.items():
        cmd = compose._compose_cmd(
            project_name,
            compose_file,
            ["exec", "-T", service, "/bin/sh", "-c", str(script_path)],
            sudo=sudo,
        )
        try:
            rc, _, err = await compose._exec(
                cmd,
                cwd=Path(compose_file).parent,
                timeout=timeout,
            )
        except TimeoutError as exc:
            raise ResetScriptError(
                f"reset script timed out for {env_name}/{service} after {timeout:g}s "
                f"(project={project_name!r}, command={shlex.join(cmd)!r})"
            ) from exc
        if rc != 0:
            detail = err.strip() or "(no stderr)"
            raise ResetScriptError(
                f"reset script failed for {env_name}/{service} with exit code {rc} "
                f"(project={project_name!r}, command={shlex.join(cmd)!r}): {detail}"
            )


async def reset_environment(
    env_name: str,
    ports: dict[str, int],
    env_config: dict[str, Any],
    *,
    project_name: str | None = None,
    compose_file: str | Path | None = None,
    sudo: bool | None = None,
    endpoint_timeout: float = 30,
    script_timeout: float = 60,
    max_retries: int = 10,
) -> None:
    """Reset *env_name*: endpoints first, scripts as fallback. No-op if neither configured."""
    env_def = (env_config.get("environments") or {}).get(env_name, {})
    reset_endpoints = env_def.get("reset_endpoints") or {}
    reset_scripts = env_def.get("reset_scripts") or {}
    if not reset_endpoints and not reset_scripts:
        return

    if reset_endpoints:
        try:
            await reset_via_endpoints(
                env_name,
                ports,
                env_config,
                timeout=endpoint_timeout,
                max_retries=max_retries,
            )
            return
        except RuntimeError:
            if not (reset_scripts and project_name and compose_file):
                raise

    if reset_scripts:
        if not project_name or not compose_file:
            raise RuntimeError(
                f"reset scripts configured for {env_name} but project_name/compose_file missing"
            )
        await reset_via_scripts(
            env_name,
            project_name,
            compose_file,
            env_config,
            sudo=sudo,
            timeout=script_timeout,
        )


__all__ = [
    "ResetScriptError",
    "render_template",
    "reset_via_endpoints",
    "reset_via_scripts",
    "reset_environment",
]
