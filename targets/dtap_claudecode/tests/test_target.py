"""Offline tests for DtapClaudeCodeTarget's host-side hooks and helpers
(no Docker, no SDK)."""

from __future__ import annotations

import json
import os

import pytest
from dtap_scaffold.types import AgentLaunchSpec, EpisodeResult

from dtap_claudecode_target import DtapClaudeCodeTarget
from dtap_claudecode_target.target import (
    CONTAINER_MOUNT,
    CONTAINER_WORKSPACE,
    OS_FILESYSTEM_DISALLOWED_TOOLS,
)
from dtap_claudecode_target.trajectory import RESULT_FILENAME, TRANSCRIPT_FILENAME


def _target(**kw) -> DtapClaudeCodeTarget:
    return DtapClaudeCodeTarget(model="claude-x", api_base="http://proxy", api_key="k", **kw)


def _spec(**overrides) -> AgentLaunchSpec:
    base = dict(
        model="claude-x",
        api_base="http://proxy",
        api_key="secret",
        system_prompt="You are an agent.",
        instructions=("do the task",),
        proxy_url="http://host.docker.internal:9000/mcp",
        mcp_server_names=("travel-suite",),
        skills=(),
        native_tool_deny=("Bash",),
        max_turns=42,
        temperature=None,
        workspace_dir=None,
        output_dir=None,
        metadata={"task_dir": "/data/travel/1", "domain": "travel"},
    )
    base.update(overrides)
    return AgentLaunchSpec(**base)


# ----- _agent_kind / _native_tool_deny ------------------------------------


def test_agent_kind():
    assert _target()._agent_kind() == "claude_code"


def test_native_tool_deny_enabled_is_empty():
    assert _target()._native_tool_deny("enabled") == []
    assert _target()._native_tool_deny("") == []  # default


def test_native_tool_deny_disabled_is_upstream_list():
    deny = _target()._native_tool_deny("disabled")
    assert deny == list(OS_FILESYSTEM_DISALLOWED_TOOLS)
    assert "Bash" in deny and "Read" in deny and "AskUserQuestion" in deny


def test_native_tool_deny_custom_json_list():
    assert _target()._native_tool_deny('["Bash", "Write"]') == ["Bash", "Write"]


def test_native_tool_deny_non_list_json_raises():
    with pytest.raises(ValueError):
        _target()._native_tool_deny('{"Bash": true}')


def test_native_tool_deny_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        _target()._native_tool_deny("Bash,Write")  # not 'enabled'/'disabled', not JSON


# ----- _build_task --------------------------------------------------------


def test_build_task_payload():
    t = _target()
    task = t._build_task(_spec(), output_dir=CONTAINER_MOUNT, workspace_dir="/dtap/workspace")
    assert task["model"] == "claude-x"
    assert task["system_prompt"] == "You are an agent."
    assert task["instructions"] == ["do the task"]
    assert task["proxy_url"] == "http://host.docker.internal:9000/mcp"
    assert task["native_tool_deny"] == ["Bash"]
    assert task["max_turns"] == 42
    assert task["output_dir"] == CONTAINER_MOUNT
    assert task["workspace_dir"] == "/dtap/workspace"
    assert task["metadata"]["domain"] == "travel"
    # JSON-serializable
    json.dumps(task)


def test_build_task_includes_skills():
    """The DTAP skill injection vector must reach the container payload: _build_task carries
    spec.skills into task["skills"], which the in-container driver.materialize_skills consumes.
    spec.skills itself is set by the agent-agnostic base (_precall_skills) and tested there; this
    guards the claudecode-specific bridge, whose silent drop a green offline suite would otherwise
    miss (driver.run_episode is container-only / no-cover)."""
    t = _target()
    skills = ({"name": "helper", "content": "SKILL BODY", "mode": "create", "row": 1},)
    task = t._build_task(_spec(skills=skills), output_dir=CONTAINER_MOUNT, workspace_dir="/w")
    assert task["skills"] == [dict(skills[0])]  # the injected skill dict reaches the payload
    json.dumps(task)  # and stays JSON-serializable for the container boundary


# ----- _docker_command ----------------------------------------------------


def test_docker_command_wiring():
    t = _target(image="my-image:1")
    cmd = t._docker_command(_spec(), "/host/instance")
    assert cmd[0:3] == ["docker", "run", "--rm"]
    assert "--add-host" in cmd
    i = cmd.index("--add-host")
    assert cmd[i + 1] == "host.docker.internal:host-gateway"
    assert "-e" in cmd and "ANTHROPIC_BASE_URL=http://proxy" in cmd
    # the token rides the subprocess env (name-only -e), NEVER the argv/ps output
    assert "ANTHROPIC_AUTH_TOKEN" in cmd
    assert not any(c.startswith("ANTHROPIC_AUTH_TOKEN=") for c in cmd)  # value not on argv
    assert "secret" not in cmd
    assert "ANTHROPIC_MODEL=claude-x" in cmd
    assert f"DTAP_TASK_FILE={CONTAINER_MOUNT}/task.json" in cmd
    assert f"/host/instance:{CONTAINER_MOUNT}" in cmd
    assert cmd[-1] == "my-image:1"


def test_docker_command_omits_absent_credentials():
    t = DtapClaudeCodeTarget(model="m")  # no api_base/api_key
    cmd = t._docker_command(_spec(api_base=None, api_key=None, model="m"), "/i")
    assert not any(c.startswith("ANTHROPIC_BASE_URL=") for c in cmd)
    assert not any(c.startswith("ANTHROPIC_AUTH_TOKEN=") for c in cmd)
    assert "ANTHROPIC_MODEL=m" in cmd


# ----- _read_result -------------------------------------------------------


def test_read_result_ok(tmp_path):
    with open(os.path.join(str(tmp_path), RESULT_FILENAME), "w") as fh:
        json.dump({"final_output": "done", "error": None, "duration": 1.5}, fh)
    assert _target()._read_result(str(tmp_path)) == ("done", None, 1.5)


def test_read_result_missing_degrades(tmp_path):
    assert _target()._read_result(str(tmp_path)) == ("", None, 0.0)


def test_read_result_error_and_bad_duration(tmp_path):
    with open(os.path.join(str(tmp_path), RESULT_FILENAME), "w") as fh:
        json.dump({"final_output": "", "error": "boom", "duration": "nope"}, fh)
    final, err, dur = _target()._read_result(str(tmp_path))
    assert final == "" and err == "boom" and dur == 0.0


def test_read_result_garbled_degrades(tmp_path):
    """result.json sits on the /dtap bind mount the agent shares; with native tools
    default-enabled the attacker-influenced agent can plant arbitrary bytes there (and the
    driver's final overwrite can lose the teardown race), so the reader must tolerate a
    hostile file. Each shape -- a non-dict top level, invalid UTF-8, an oversized int, and
    deeply-nested JSON -- degrades to empty outputs rather than raise out of run() and
    abandon the whole task (each would raise against the pre-hardening reader)."""
    path = os.path.join(str(tmp_path), RESULT_FILENAME)
    for raw in (
        b"[1, 2, 3]",  # non-dict list -> pre-guard data.get() would AttributeError
        b"42",  # non-dict scalar
        b"\xff\xfe\x00bad",  # invalid utf-8 -> UnicodeDecodeError (a ValueError)
        b"1" + b"0" * 4400,  # 4401-digit int -> bare ValueError (int_max_str_digits limit)
        b"[" * 100000,  # deeply-nested -> RecursionError
    ):
        with open(path, "wb") as fh:
            fh.write(raw)
        assert _target()._read_result(str(tmp_path)) == ("", None, 0.0)


def test_read_result_oversized_duration_degrades(tmp_path):
    """A hostile result.json can carry a "duration" that is a parse-valid but
    float-overflowing integer: >~308 digits (float() -> OverflowError) yet under json's
    4300-digit cap, so it survives json.load AND the isinstance(dict) guard and reaches
    float(). OverflowError is an ArithmeticError, NOT a ValueError, so the narrow
    (TypeError, ValueError) catch missed it and it raised out of run() -- abandoning the
    whole task. The duration must degrade to 0.0 while the other fields survive (this
    raises OverflowError against the pre-fix reader, so the test is non-vacuous)."""
    path = os.path.join(str(tmp_path), RESULT_FILENAME)
    big = "1" + "0" * 399  # 400-digit int: parses, dict-valued, but float() overflows
    with open(path, "w") as fh:
        fh.write('{"final_output": "done", "error": null, "duration": ' + big + "}")
    assert _target()._read_result(str(tmp_path)) == ("done", None, 0.0)


# ----- _run_episode + _extract_trajectory via a faked _docker_run ---------


async def test_run_episode_and_extract(tmp_path, monkeypatch):
    t = _target()

    instance = tmp_path / "inst"
    instance.mkdir()
    # canned transcript + result the faked docker run "produces"
    with open(instance / TRANSCRIPT_FILENAME, "w") as fh:
        fh.write(json.dumps({"record": "trace_start", "trace_id": "z", "metadata": {}}) + "\n")
        fh.write(json.dumps({"record": "user_input", "content": "go"}) + "\n")
        fh.write(
            json.dumps(
                {
                    "record": "message",
                    "message": {
                        "type": "assistant",
                        "content": [{"type": "text", "text": "ANSWER"}],
                    },
                }
            )
            + "\n"
        )
        fh.write(json.dumps({"record": "trace_end", "trace_id": "z"}) + "\n")
    with open(instance / RESULT_FILENAME, "w") as fh:
        json.dump({"final_output": "ANSWER", "error": None, "duration": 0.3}, fh)

    async def fake_docker_run(spec):
        return str(instance)

    monkeypatch.setattr(t, "_docker_run", fake_docker_run)

    episode = await t._run_episode(_spec())
    assert isinstance(episode, EpisodeResult)
    assert episode.output_dir == str(instance)
    assert episode.final_output == "ANSWER"
    assert episode.duration == 0.3

    art = t._extract_trajectory(episode)
    assert art.final_response == "ANSWER"
    assert art.agent_responses == ("ANSWER",)


# ----- _exec_on_host (host_code_execution Docker seam) --------------------


async def test_exec_on_host_builds_docker_cmd(tmp_path, monkeypatch):
    """The host_code_execution seam runs attacker code in the AGENT image with the
    run workspace bind-mounted at the agent workspace path (entrypoint overridden to
    ``sh``, IS_SANDBOX=1), and returns the combined stdout/stderr for the next round."""
    import asyncio

    t = _target()
    t._run_dir = str(tmp_path)
    captured: dict = {}

    class _FakeProc:
        async def communicate(self):
            return b"ran-output", None

    async def fake_exec(*cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    out = await t._exec_on_host("echo hi > f.txt")

    assert out == "ran-output"
    cmd = captured["cmd"]
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--entrypoint" in cmd and "sh" in cmd
    assert "IS_SANDBOX=1" in cmd  # bypassPermissions-as-root seam, same as the episode
    workspace = os.path.join(str(tmp_path), "workspace")
    assert f"{workspace}:{CONTAINER_WORKSPACE}" in cmd  # attacker + agent share the workspace
    assert cmd[-3:] == [t._image, "-c", "echo hi > f.txt"]
