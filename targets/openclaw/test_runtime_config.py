"""Unit tests for the grounded gateway config + runtime command builders.

These cover the deterministic pieces (config JSON, state-dir materialization,
local + Docker command/env construction, capability registry) without needing a
real OpenClaw gateway or Docker daemon.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openclaw_target.config import (
    DEFAULT_PLUGIN_NAME,
    build_gateway_config,
    materialize_state_dir,
)
from openclaw_target.docker_runtime import DEFAULT_DOCKER_IMAGE, OpenClawDockerRuntime
from openclaw_target.runtime import OpenClawRuntime
from openclaw_target.target import (
    MESSAGE_CONTENT_CTRL,
    SHELL_OUTPUT_CTRL,
    TOOL_OUTPUT_CONTROLLABLES,
    OpenClawTarget,
)

# -- config builder -----------------------------------------------------------


def test_minimal_config_is_empty() -> None:
    assert build_gateway_config() == {}


def test_config_provider_routing_grounded() -> None:
    cfg = build_gateway_config(
        model_id="openai/gpt-5",
        provider_base_url="http://host.docker.internal:9001",
        provider_api_key="sk-test",
        tool_policy="minimal",
        workspace_dir="/home/node/.openclaw/workspace",
        plugin_names=[DEFAULT_PLUGIN_NAME],
    )

    provider = cfg["models"]["providers"]["openai"]
    # /v1 is appended (OpenClaw calls <baseUrl>/chat/completions).
    assert provider["baseUrl"] == "http://host.docker.internal:9001/v1"
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "sk-test"
    # allowPrivateNetwork is required to reach a loopback / host.docker.internal proxy.
    assert provider["request"]["allowPrivateNetwork"] is True
    assert provider["models"] == [{"id": "gpt-5", "name": "gpt-5"}]
    assert cfg["models"]["mode"] == "replace"

    assert cfg["agents"]["defaults"]["model"] == {"primary": "openai/gpt-5"}
    assert cfg["agents"]["defaults"]["workspace"] == "/home/node/.openclaw/workspace"
    assert cfg["agents"]["list"][0]["id"] == "main"
    assert cfg["tools"] == {"profile": "minimal"}
    assert cfg["plugins"] == {"enabled": True, "allow": [DEFAULT_PLUGIN_NAME]}


def test_config_does_not_double_append_v1() -> None:
    cfg = build_gateway_config(
        model_id="openai/gpt-5",
        provider_base_url="http://127.0.0.1:9001/v1",
    )
    assert cfg["models"]["providers"]["openai"]["baseUrl"] == "http://127.0.0.1:9001/v1"


def test_config_unqualified_model_uses_synthetic_provider() -> None:
    cfg = build_gateway_config(model_id="gpt-5", provider_base_url="http://x/v1")
    assert "superred" in cfg["models"]["providers"]
    assert cfg["models"]["providers"]["superred"]["models"] == [
        {"id": "gpt-5", "name": "gpt-5"},
    ]


def test_provider_model_entry_includes_required_name() -> None:
    """OpenClaw rejects models[] rows with id-only (models.0.name required)."""
    cfg = build_gateway_config(
        model_id="openai/gpt-4o-mini",
        provider_base_url="http://127.0.0.1:9001/v1",
    )
    assert cfg["models"]["providers"]["openai"]["models"] == [
        {"id": "gpt-4o-mini", "name": "gpt-4o-mini"},
    ]


def test_materialize_state_dir(tmp_path: Path) -> None:
    plugin_src = tmp_path / "plugin"
    plugin_src.mkdir()
    (plugin_src / "index.js").write_text("// plugin")
    (plugin_src / "openclaw.plugin.json").write_text('{"id": "superred-injection"}')

    state = tmp_path / "state"
    cfg = build_gateway_config(model_id="openai/gpt-5", provider_base_url="http://x")
    materialize_state_dir(state, config=cfg, plugin_src=plugin_src, plugin_name="superred-injection")

    written = json.loads((state / "openclaw.json").read_text())
    assert written["models"]["mode"] == "replace"
    assert (state / "workspace").is_dir()
    ext = state / "extensions" / "superred-injection"
    assert ext.joinpath("index.js").read_text() == "// plugin"
    assert ext.joinpath("openclaw.plugin.json").read_text().startswith("{")


# -- local runtime ------------------------------------------------------------


def test_local_runtime_cmd_binds_loopback() -> None:
    rt = OpenClawRuntime(host_port=12345)
    cmd = rt._build_cmd()
    assert cmd == ["openclaw", "gateway", "--bind", "loopback", "--port", "12345", "--allow-unconfigured"]


def test_local_runtime_env_uses_state_dir_not_fabricated_vars(tmp_path: Path) -> None:
    rt = OpenClawRuntime(
        state_dir=str(tmp_path),
        model_id="openai/gpt-5",
        provider_base_url="http://127.0.0.1:9001",
        callback_url="http://127.0.0.1:8899",
    )
    rt._auth_token = "tok"
    rt._prepare_state_dir()
    env = rt._build_env()

    assert env["OPENCLAW_STATE_DIR"] == str(tmp_path)
    assert env["OPENCLAW_CONFIG_PATH"] == str(tmp_path / "openclaw.json")
    assert env["OPENCLAW_GATEWAY_TOKEN"] == "tok"
    assert env["SUPERRED_CALLBACK_URL"] == "http://127.0.0.1:8899"
    # The previously-fabricated env vars must NOT be used; provider routing is
    # config (openclaw.json), extensions live under <stateDir>/extensions.
    assert "OPENCLAW_PROVIDER_URL" not in env
    assert "OPENCLAW_EXTENSIONS_DIR" not in env
    # Provider routing landed in config instead.
    cfg = json.loads((tmp_path / "openclaw.json").read_text())
    assert cfg["models"]["providers"]["openai"]["baseUrl"] == "http://127.0.0.1:9001/v1"


# -- docker runtime -----------------------------------------------------------


def test_docker_default_image_is_official_release() -> None:
    rt = OpenClawDockerRuntime()
    assert rt.image == DEFAULT_DOCKER_IMAGE
    assert rt.image.startswith("ghcr.io/openclaw/openclaw:")


@pytest.mark.asyncio
async def test_ensure_image_pulls_when_missing() -> None:
    rt = OpenClawDockerRuntime(image="ghcr.io/openclaw/openclaw:latest", auto_pull_image=True)
    calls: list[list[str]] = []

    async def fake_docker(args: list[str], *, check: bool) -> tuple[int, str]:
        calls.append(args)
        if args[:2] == ["image", "inspect"]:
            return 1, ""
        if args[0] == "pull":
            return 0, ""
        raise AssertionError(f"unexpected docker call: {args}")

    rt._docker = fake_docker  # type: ignore[method-assign]
    await rt._ensure_image()
    assert calls == [
        ["image", "inspect", "ghcr.io/openclaw/openclaw:latest"],
        ["pull", "ghcr.io/openclaw/openclaw:latest"],
    ]


@pytest.mark.asyncio
async def test_ensure_image_skips_pull_when_present() -> None:
    rt = OpenClawDockerRuntime(image="ghcr.io/openclaw/openclaw:latest")
    calls: list[list[str]] = []

    async def fake_docker(args: list[str], *, check: bool) -> tuple[int, str]:
        calls.append(args)
        return 0, ""

    rt._docker = fake_docker  # type: ignore[method-assign]
    await rt._ensure_image()
    assert calls == [["image", "inspect", "ghcr.io/openclaw/openclaw:latest"]]


@pytest.mark.asyncio
async def test_ensure_image_raises_when_missing_and_auto_pull_disabled() -> None:
    rt = OpenClawDockerRuntime(
        image="ghcr.io/openclaw/openclaw:missing",
        auto_pull_image=False,
    )

    async def fake_docker(args: list[str], *, check: bool) -> tuple[int, str]:
        return 1, ""

    rt._docker = fake_docker  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="auto_pull_image=False"):
        await rt._ensure_image()


@pytest.mark.asyncio
async def test_seed_state_volume_uses_docker_cp_not_bind_mount(tmp_path: Path) -> None:
    """The seed path must go through ``docker cp`` (portable across Docker
    VMs / remote daemons), not a host bind mount of the source dir."""
    rt = OpenClawDockerRuntime(image="openclaw:local", container_name="cname")
    calls: list[list[str]] = []

    async def fake_docker(args: list[str], *, check: bool) -> tuple[int, str]:
        calls.append(args)
        return 0, ""

    rt._docker = fake_docker  # type: ignore[method-assign]
    volume = await rt._seed_state_volume(tmp_path)

    assert volume == "cname-state"
    assert calls[0] == ["volume", "create", "cname-state"]
    run_call = calls[1]
    assert run_call[:3] == ["run", "-d", "--name"]
    assert "cname-seed" in run_call
    assert "--user" in run_call and "root" in run_call
    assert "cname-state:/home/node/.openclaw" in run_call
    # No host directory should ever be bind-mounted for the seed step.
    assert not any(str(tmp_path) in arg for arg in run_call)
    cp_call = calls[2]
    assert cp_call[0] == "cp"
    assert cp_call[1] == f"{tmp_path}/."
    assert cp_call[2] == "cname-seed:/home/node/.openclaw"
    chown_call = calls[3]
    assert chown_call[:3] == ["exec", "--user", "root"]
    assert "chown" in chown_call and "node:node" in chown_call
    assert calls[-1] == ["rm", "-f", "cname-seed"]


@pytest.mark.asyncio
async def test_seed_state_volume_cleans_up_volume_on_failure(tmp_path: Path) -> None:
    rt = OpenClawDockerRuntime(image="openclaw:local", container_name="cname")
    calls: list[list[str]] = []

    async def fake_docker(args: list[str], *, check: bool) -> tuple[int, str]:
        calls.append(args)
        if args[0] == "cp":
            if check:
                raise RuntimeError("docker cp failed: boom")
            return 1, "boom"
        return 0, ""

    rt._docker = fake_docker  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="docker cp failed"):
        await rt._seed_state_volume(tmp_path)

    assert ["volume", "rm", "-f", "cname-state"] in calls
    assert ["rm", "-f", "cname-seed"] in calls


def test_docker_run_cmd_grounded(tmp_path: Path) -> None:
    rt = OpenClawDockerRuntime(
        image="openclaw:local",
        host_port=20001,
        callback_url="http://host.docker.internal:8899",
        container_name="superred-openclaw-test",
    )
    rt._auth_token = "tok"
    rt._state_path = tmp_path
    rt._volume_name = "superred-openclaw-test-state"
    cmd = rt._build_run_cmd()
    joined = " ".join(cmd)

    assert cmd[:3] == ["docker", "run", "-d"]
    # Dynamic host port published to the in-container gateway port.
    assert "-p" in cmd and "20001:18789" in cmd
    # Container reaches host services via host.docker.internal.
    assert "host.docker.internal:host-gateway" in cmd
    # Non-loopback bind => token is mandatory; passed via env.
    assert "OPENCLAW_GATEWAY_TOKEN=tok" in cmd
    assert "SUPERRED_CALLBACK_URL=http://host.docker.internal:8899" in cmd
    # Seeded named volume (not a raw host bind - see _seed_state_volume) is
    # mounted into the pinned container path.
    assert "superred-openclaw-test-state:/home/node/.openclaw" in cmd
    # Hardening from compose.
    assert "no-new-privileges:true" in cmd
    # Gateway launched bound to lan on the container port.
    assert joined.endswith("openclaw gateway --bind lan --port 18789 --allow-unconfigured")
    assert cmd[cmd.index("openclaw:local") + 1] == "openclaw"


def test_docker_run_cmd_includes_callback_token(tmp_path: Path) -> None:
    rt = OpenClawDockerRuntime(
        image="openclaw:local",
        host_port=20001,
        callback_url="http://host.docker.internal:8899",
        callback_token="cb-tok",
    )
    rt._auth_token = "tok"
    cmd = rt._build_run_cmd()
    assert "SUPERRED_CALLBACK_TOKEN=cb-tok" in cmd


def test_local_runtime_env_includes_callback_token(tmp_path: Path) -> None:
    rt = OpenClawRuntime(
        state_dir=str(tmp_path),
        callback_url="http://127.0.0.1:8899",
        callback_token="cb-tok",
    )
    rt._auth_token = "tok"
    rt._prepare_state_dir()
    env = rt._build_env()
    assert env["SUPERRED_CALLBACK_TOKEN"] == "cb-tok"


def test_docker_runtime_gateway_url_uses_host_port() -> None:
    rt = OpenClawDockerRuntime(host_port=20002)
    assert rt.gateway_url == "ws://127.0.0.1:20002"
    assert rt.container_host == "host.docker.internal"


# -- capability registry (Phase 5b) -------------------------------------------


def test_registry_has_grounded_capabilities() -> None:
    assert TOOL_OUTPUT_CONTROLLABLES["exec"] is SHELL_OUTPUT_CTRL
    assert TOOL_OUTPUT_CONTROLLABLES["process"] is SHELL_OUTPUT_CTRL
    assert TOOL_OUTPUT_CONTROLLABLES["message"] is MESSAGE_CONTENT_CTRL
    # memory is a plugin slot, not a tool — must not be a fabricated tool entry.
    assert "memory" not in TOOL_OUTPUT_CONTROLLABLES
    # bash is not an agent-catalogue tool (it is a sessions-SDK / ACP surface),
    # so it must not be wired as a tool-output injection point.
    assert "bash" not in TOOL_OUTPUT_CONTROLLABLES


def test_get_controllables_includes_new_capabilities() -> None:
    target = OpenClawTarget(enable_tool_injection=True)
    names = {c.name for c in target.get_controllables()}
    assert {"shell_output", "message_content", "web_content", "file_content"} <= names
    assert "memory_poison" in names


def test_get_controllables_dedupes() -> None:
    target = OpenClawTarget(enable_tool_injection=True)
    ctrls = target.get_controllables()
    # exec/process both map to the single SHELL_OUTPUT_CTRL.
    assert sum(1 for c in ctrls if c.name == "shell_output") == 1


def test_docker_mode_uses_host_alias() -> None:
    target = OpenClawTarget(managed=True, managed_runtime="docker")
    assert target._is_docker is True
    assert target._container_host() == "host.docker.internal"
    assert target._bind_host() == "0.0.0.0"


def test_local_mode_uses_loopback() -> None:
    target = OpenClawTarget(managed=True, managed_runtime="local")
    assert target._is_docker is False
    assert target._container_host() == "127.0.0.1"
    assert target._bind_host() == "127.0.0.1"
