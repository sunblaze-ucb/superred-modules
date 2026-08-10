"""The episode container must be named, and removed when the backstop fires."""

from __future__ import annotations

import subprocess

import pytest
from dtap_openclaw_target import driver


def test_container_is_named_so_it_can_be_removed(monkeypatch, tmp_path) -> None:
    seen: list[list[str]] = []

    def fake_run_docker(cmd, timeout):
        seen.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(driver, "_run_docker", fake_run_docker)
    monkeypatch.setattr(driver, "write_episode_inputs", lambda *a, **k: None)

    episode = tmp_path / "episode-abc123"
    driver.run_openclaw_container(
        _spec(), image="img", timeout=1.0, episode_dir=str(episode)
    )

    cmd = seen[0]
    assert "--name" in cmd, "container must be named or a timeout cannot remove it"
    assert cmd[cmd.index("--name") + 1] == driver._container_name(str(episode))


def test_timed_out_container_is_removed(monkeypatch, tmp_path) -> None:
    """subprocess's timeout kills only the docker client.

    A surviving container keeps issuing tools/call at the host MCP proxy, which
    by then has been rebound to the next run.
    """
    removed: list[str] = []

    def fake_run_docker(cmd, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(driver, "_run_docker", fake_run_docker)
    monkeypatch.setattr(driver, "write_episode_inputs", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_remove_container", removed.append)

    episode = tmp_path / "episode-def456"
    out = driver.run_openclaw_container(
        _spec(), image="img", timeout=0.01, episode_dir=str(episode)
    )

    # Still non-fatal: the partial trace is returned and the run is judged.
    assert out == str(episode)
    assert removed == [driver._container_name(str(episode))]


def _spec():
    from dtap_scaffold.agent_base import AgentLaunchSpec

    return AgentLaunchSpec(
        model="m",
        api_base=None,
        api_key=None,
        system_prompt="You are a travel agent.",
        instructions=("Book a flight.",),
        proxy_url="http://host.docker.internal:9999/mcp",
        mcp_server_names=("travel-suite",),
        workspace_dir="/state",
        output_dir=None,
    )
