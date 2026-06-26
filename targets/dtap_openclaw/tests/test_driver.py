"""Offline tests for the OpenClaw Docker driver.

No Docker is run: the pure config/agents/task builders are checked directly, and
``run_openclaw_container`` is exercised with the single Docker boundary
(``driver._run_docker``) monkeypatched, so file materialization and the
``docker run`` command are verified without a daemon.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dtap_scaffold.types import AgentLaunchSpec

from dtap_openclaw_target import driver


def _spec(**over) -> AgentLaunchSpec:
    base = dict(
        model="gpt-4o-2024-05-13",
        api_base="http://proxy:4000",
        api_key="sk-test",
        system_prompt="You are a travel agent.",
        instructions=("Book a flight.", "Confirm it."),
        proxy_url="http://host.docker.internal:9999/mcp",
        mcp_server_names=("travel-suite", "gmail"),
        skills=(),
        native_tool_deny=(),
        max_turns=42,
        temperature=None,
        workspace_dir="/state",
        output_dir=None,
        metadata={},
    )
    base.update(over)
    return AgentLaunchSpec(**base)


# --- mcp_server_url --------------------------------------------------------


def test_mcp_server_url_appends_server_segment() -> None:
    assert (
        driver.mcp_server_url("http://host.docker.internal:9999/mcp", "travel-suite")
        == "http://host.docker.internal:9999/mcp/travel-suite"
    )
    # trailing slash normalized
    assert driver.mcp_server_url("http://h/mcp/", "g") == "http://h/mcp/g"


# --- build_openclaw_config -------------------------------------------------


def test_config_provider_wired_to_litellm_proxy() -> None:
    cfg = driver.build_openclaw_config(_spec())
    provider = cfg["models"]["providers"]["litellm"]
    assert provider["baseUrl"] == "http://proxy:4000"
    assert provider["apiKey"] == "sk-test"
    assert provider["api"] == "openai-completions"  # default
    assert provider["models"][0]["id"] == "gpt-4o-2024-05-13"
    assert cfg["agents"]["defaults"]["model"]["primary"] == "litellm/gpt-4o-2024-05-13"


def test_config_provider_api_override() -> None:
    cfg = driver.build_openclaw_config(_spec(), provider_api="anthropic-messages")
    assert cfg["models"]["providers"]["litellm"]["api"] == "anthropic-messages"


def test_config_mcp_servers_one_entry_per_active_server() -> None:
    cfg = driver.build_openclaw_config(_spec())
    servers = cfg["mcp"]["servers"]
    assert set(servers) == {"travel-suite", "gmail"}
    assert servers["travel-suite"] == {
        "transport": "streamable-http",
        "url": "http://host.docker.internal:9999/mcp/travel-suite",
    }


def test_config_native_tools_enabled_by_default() -> None:
    # Upstream DTAP disabled native tools; this port ENABLES exec/fs at full.
    cfg = driver.build_openclaw_config(_spec())
    assert cfg["tools"]["exec"] == {"security": "full", "ask": "off"}
    assert cfg["tools"]["fs"] == {"security": "full", "ask": "off"}
    assert "deny" not in cfg["tools"]  # nothing denied when policy enabled
    assert cfg["tools"]["web"]["search"]["enabled"] is False
    assert cfg["browser"]["enabled"] is False


def test_config_native_tool_deny_list_applied() -> None:
    cfg = driver.build_openclaw_config(_spec(native_tool_deny=("fs", "exec", "")))
    assert cfg["tools"]["deny"] == ["exec", "fs"]  # sorted, empty dropped


def test_config_temperature_only_when_set() -> None:
    assert "temperature" not in driver.build_openclaw_config(_spec())["agents"]["defaults"]
    cfg = driver.build_openclaw_config(_spec(temperature=0.5))
    assert cfg["agents"]["defaults"]["temperature"] == 0.5


def test_config_skills_dir_added_when_skills_present() -> None:
    spec = _spec(skills=({"name": "evil", "content": "x", "mode": "create"},))
    cfg = driver.build_openclaw_config(spec, skills_dir="/state/skills")
    assert cfg["skills"]["load"]["extraDirs"] == ["/state/skills"]
    # no skills_dir -> no skills block even if skills present
    assert "skills" not in driver.build_openclaw_config(spec)


def test_config_no_mcp_block_when_no_servers() -> None:
    assert "mcp" not in driver.build_openclaw_config(_spec(mcp_server_names=()))


# --- build_agents_md / build_task_json -------------------------------------


def test_build_agents_md_is_system_prompt() -> None:
    assert driver.build_agents_md(_spec()) == "You are a travel agent."
    assert driver.build_agents_md(_spec(system_prompt="")) == ""


def test_build_task_json() -> None:
    tj = driver.build_task_json(_spec(), session_id="s1", profile="p1", thinking="high")
    assert tj["turns"] == ["Book a flight.", "Confirm it."]
    assert tj["session_id"] == "s1"
    assert tj["profile"] == "p1"
    assert tj["thinking"] == "high"
    assert tj["max_turns"] == 42
    assert tj["config_path"] == "/state/.openclaw-p1/openclaw.json"


def test_build_task_json_empty_instructions_gets_one_blank_turn() -> None:
    tj = driver.build_task_json(_spec(instructions=()), session_id="s", profile="p", thinking="low")
    assert tj["turns"] == [""]


# --- write_episode_inputs --------------------------------------------------


def test_write_episode_inputs_materializes_all_files(tmp_path) -> None:
    spec = _spec(skills=({"name": "skillA", "content": "DO BAD", "mode": "create"},))
    state = tmp_path / "state"
    paths = driver.write_episode_inputs(
        spec, str(state), session_id="s1", profile="p1", thinking="medium"
    )

    config = json.loads(Path(paths["config"]).read_text())
    assert paths["config"].endswith(".openclaw-p1/openclaw.json")
    assert config["mcp"]["servers"]["travel-suite"]["transport"] == "streamable-http"
    assert config["skills"]["load"]["extraDirs"] == ["/state/skills"]

    assert Path(paths["agents_md"]).read_text() == "You are a travel agent."
    assert Path(paths["agents_md"]).parent.name == "workspace"

    task = json.loads(Path(paths["task_json"]).read_text())
    assert task["turns"] == ["Book a flight.", "Confirm it."]

    skill_md = state / "skills" / "skillA" / "SKILL.md"
    assert skill_md.read_text() == "DO BAD"
    assert (state / "traces").is_dir()


def test_write_episode_inputs_provider_api_threaded(tmp_path) -> None:
    paths = driver.write_episode_inputs(
        _spec(),
        str(tmp_path / "s"),
        session_id="s",
        profile="p",
        thinking="low",
        provider_api="anthropic-messages",
    )
    config = json.loads(Path(paths["config"]).read_text())
    assert config["models"]["providers"]["litellm"]["api"] == "anthropic-messages"


# --- run_openclaw_container (Docker boundary monkeypatched) -----------------


def test_run_container_builds_command_and_returns_episode_dir(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def fake_run_docker(cmd, timeout):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        return 0, "ok", ""

    monkeypatch.setattr(driver, "_run_docker", fake_run_docker)

    out = driver.run_openclaw_container(
        _spec(output_dir=str(tmp_path)),
        image="img:tag",
        timeout=123.0,
        thinking="high",
        provider_api="anthropic-messages",
    )

    # per-episode subdir under output_dir
    assert out.startswith(str(tmp_path))
    assert Path(out).name.startswith("episode-")

    cmd = captured["cmd"]
    assert cmd[0:3] == ["docker", "run", "--rm"]
    assert "-v" in cmd and f"{out}:/state" in cmd
    assert "-e" in cmd and "HOME=/state" in cmd
    assert "--add-host" in cmd and "host.docker.internal:host-gateway" in cmd
    assert cmd[-1] == "img:tag"
    assert captured["timeout"] == 123.0

    # inputs were written into the episode dir, with provider_api threaded through
    task = json.loads(Path(out, "task.json").read_text())
    config = json.loads(Path(out, ".openclaw-" + task["profile"], "openclaw.json").read_text())
    assert config["models"]["providers"]["litellm"]["api"] == "anthropic-messages"
    assert task["thinking"] == "high"


def test_run_container_uses_network_when_given(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def fake_run_docker(cmd, timeout):
        captured["cmd"] = cmd
        return 0, "", ""

    monkeypatch.setattr(driver, "_run_docker", fake_run_docker)
    driver.run_openclaw_container(_spec(output_dir=str(tmp_path)), network="dtap-net")
    cmd = captured["cmd"]
    assert "--network" in cmd and "dtap-net" in cmd
    assert "--add-host" not in cmd


def test_run_container_raises_on_nonzero_exit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(driver, "_run_docker", lambda cmd, timeout: (1, "", "boom failure"))
    with pytest.raises(RuntimeError, match="openclaw container exited 1"):
        driver.run_openclaw_container(_spec(output_dir=str(tmp_path)))


def test_run_container_defaults_output_dir_to_tempdir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(driver, "_run_docker", lambda cmd, timeout: (0, "", ""))
    out = driver.run_openclaw_container(_spec(output_dir=None))
    assert Path(out).is_dir()
    assert Path(out).name.startswith("episode-")
