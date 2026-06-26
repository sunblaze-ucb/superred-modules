"""Offline tests for ``DockerEnvStack`` (the per-instance Docker + MCP lifecycle).

No Docker daemon, no MCP servers, no network: the collaborator seams
(``compose.*`` / ``reset.*`` / ``env_registry.load`` / ``_spawn_process`` /
``_wait_for_ready``) are monkeypatched, and the real ``PortLeaser`` /
``make_instance_state`` run against ``tmp_path``. These tests pin the lifecycle's
ordering and the fixes found during live verification:

* ``_spawn_process(log_path=...)`` captures a crashing server's stdout+stderr
  (the missing-``ujson`` crash that previously vanished into DEVNULL);
* ``up()`` wraps a readiness timeout with those server log tails;
* ``_server_env`` inherits the parent environment (so a spawned ``python3``
  server finds ``PATH``);
* ``_READY_TIMEOUT`` is the lowered 150s, not the old 600s that masked the hang;
* the env registry loads LAZILY (a stack constructs with no SDK present).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dtap_scaffold.docker import compose
from dtap_scaffold.docker import lifecycle as lc
from dtap_scaffold.docker import reset as reset_mod


# --------------------------------------------------------------------------- #
# a fake registry: one env ("travelenv") backing one server ("travel-suite")  #
# plus one injection server. No SDK / dt_arena/config needed.                 #
# --------------------------------------------------------------------------- #


class FakeRegistry:
    def __init__(self, compose_path: Path) -> None:
        self._compose = compose_path
        self.require_calls: list[tuple[str, ...]] = []

    def require_text_only(self, active_servers):
        self.require_calls.append(tuple(active_servers))

    def active_environments(self, active_servers):
        return ["travelenv"]

    def env_ports(self, env):
        return {"TRAVEL_PORT": {"default": 8080, "container_port": 80}}

    def compose_file(self, env):
        return self._compose

    def health_timeout(self, env):
        return 120

    def mcp_server(self, server):
        if server == "travel-suite":
            return {
                "name": "travel-suite",
                "path": "travel/server.py",
                "command": ["python3", "server.py", "--port", "${TRAVEL_MCP_PORT}"],
                "env": {
                    "TRAVEL_MCP_PORT": "${TRAVEL_MCP_PORT}",
                    "API": "http://h:${TRAVEL_PORT}",
                },
            }
        return None

    def mcp_base_dir(self):
        return Path("/fake/mcp_server")

    def required_injection_servers(self, injection_config):
        if injection_config:
            return {
                "travel-injection": {
                    "name": "travel-injection",
                    "path": "travel/inject.py",
                    "command": ["python3", "inject.py"],
                    "env": {"PORT": "${PORT}"},
                }
            }
        return {}

    def injection_base_dir(self):
        return Path("/fake/injection_mcp_server")

    @staticmethod
    def server_url(name, port, host="127.0.0.1"):
        return f"http://{host}:{port}/mcp"

    @property
    def env_config(self):
        return {
            "environments": {"travelenv": {"reset_scripts": {"default": "reset.sh"}}}
        }


class _FakeProc:
    def __init__(self) -> None:
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Install all collaborator fakes; return a recorder of side effects."""
    rec: dict = {
        "compose_up": [],
        "compose_down": [],
        "setup": [],
        "reset_env": [],
        "spawned": [],
    }
    compose_path = tmp_path / "docker-compose.yml"
    compose_path.write_text("services: {}\n")
    registry = FakeRegistry(compose_path)

    monkeypatch.setattr(lc.env_registry, "load", lambda config_dir=None: registry)

    async def _needs_sudo():
        return False

    async def _compose_up(project, cf, *, ports=None, sudo=None, pull=True):
        rec["compose_up"].append((project, str(cf), dict(ports or {})))

    async def _wait_healthy(project, cf, *, sudo=None, timeout=120, interval=2.0):
        return True

    async def _compose_down(project, cf, *, sudo=None):
        rec["compose_down"].append((project, str(cf)))

    async def _exec(cmd, *, cwd=None, env=None, timeout=None):
        rec["setup"].append((list(cmd), cwd))
        return (0, "", "")

    async def _reset_env(env_name, ports, env_config, **kw):
        rec["reset_env"].append((env_name, dict(ports)))

    monkeypatch.setattr(compose, "needs_sudo", _needs_sudo)
    monkeypatch.setattr(compose, "compose_up", _compose_up)
    monkeypatch.setattr(compose, "wait_healthy", _wait_healthy)
    monkeypatch.setattr(compose, "compose_down", _compose_down)
    monkeypatch.setattr(compose, "_exec", _exec)
    monkeypatch.setattr(reset_mod, "reset_environment", _reset_env)

    def _spawn(cmd, *, cwd=None, env=None, log_path=None):
        proc = _FakeProc()
        rec["spawned"].append(
            {"cmd": list(cmd), "cwd": cwd, "env": dict(env or {}), "log": log_path}
        )
        return proc

    monkeypatch.setattr(lc, "_spawn_process", _spawn)

    async def _ready_ok(urls, **kw):
        return None

    monkeypatch.setattr(lc, "_wait_for_ready", _ready_ok)
    rec["registry"] = registry
    return rec


def _stack(tmp_path, *, injection=None):
    return lc.DockerEnvStack(
        active_servers=("travel-suite",),
        injection_config=injection,
        task_dir=None,
        state_root=str(tmp_path / "state"),
    )


# --------------------------------------------------------------------------- #
# _spawn_process: the log-capture fix (real subprocess, no fakery)            #
# --------------------------------------------------------------------------- #


def test_spawn_process_captures_stdout_and_stderr_to_log(tmp_path):
    """A crashing server's output must land in the log (not vanish into DEVNULL)."""
    log = tmp_path / "srv.log"
    proc = lc._spawn_process(
        ["python3", "-c", "import sys; print('OUT'); sys.stderr.write('ERR-ujson')"],
        log_path=str(log),
    )
    proc.wait(timeout=10)
    text = log.read_text()
    assert "OUT" in text
    assert "ERR-ujson" in text  # stderr is merged into the same log (STDOUT redirect)


def test_spawn_process_without_log_does_not_crash(tmp_path):
    proc = lc._spawn_process(["python3", "-c", "print('hi')"])
    assert proc.wait(timeout=10) == 0


# --------------------------------------------------------------------------- #
# readiness seam                                                               #
# --------------------------------------------------------------------------- #


def test_ready_timeout_is_lowered_to_150():
    # The old 600s default let a crash-on-start hang for 10 minutes.
    assert lc._READY_TIMEOUT == 150.0


async def test_wait_for_ready_empty_returns_immediately():
    await lc._wait_for_ready({})  # no servers -> no wait, no raise


async def test_wait_for_ready_raises_naming_pending(monkeypatch):
    monkeypatch.setattr(lc, "_is_listening", lambda url: False)
    with pytest.raises(RuntimeError, match="travel-suite"):
        await lc._wait_for_ready(
            {"travel-suite": "http://127.0.0.1:9/mcp"}, timeout=0.05, interval=0.01
        )


async def test_wait_for_ready_succeeds_when_listening(monkeypatch):
    monkeypatch.setattr(lc, "_is_listening", lambda url: True)
    await lc._wait_for_ready({"s": "http://127.0.0.1:9/mcp"}, timeout=1.0)


# --------------------------------------------------------------------------- #
# _server_env: inherits the parent environment (PATH fix)                      #
# --------------------------------------------------------------------------- #


def test_server_env_inherits_parent_environment(monkeypatch):
    monkeypatch.setenv("DTAP_SENTINEL_VAR", "keepme")
    cfg = {
        "env": {
            "TRAVEL_MCP_PORT": "${TRAVEL_MCP_PORT}",
            "API": "http://h:${TRAVEL_PORT}",
        }
    }
    env = lc._server_env(
        cfg, "TRAVEL_MCP_PORT", 12345, {"TRAVEL_PORT": 8080}, {"X": "y"}
    )
    assert env["PATH"] == os.environ["PATH"]  # inherited -> python3 is findable
    assert env["DTAP_SENTINEL_VAR"] == "keepme"
    assert env["TRAVEL_MCP_PORT"] == "12345"  # own listen port
    assert env["API"] == "http://h:8080"  # rendered from container_ports
    assert env["X"] == "y"  # extra overrides win


def test_expand_command_expands_braced_and_bare_vars():
    cfg = {"command": ["run", "--port", "${PORT}", "-h", "$HOST"]}
    out = lc._expand_command(cfg, {"PORT": "7000", "HOST": "localhost"})
    assert out == ["run", "--port", "7000", "-h", "localhost"]


# --------------------------------------------------------------------------- #
# lazy registry (construct without SDK)                                        #
# --------------------------------------------------------------------------- #


def test_construction_does_not_load_registry(monkeypatch, tmp_path):
    calls = {"n": 0}

    def _load(config_dir=None):
        calls["n"] += 1
        return FakeRegistry(tmp_path / "c.yml")

    monkeypatch.setattr(lc.env_registry, "load", _load)
    stack = _stack(tmp_path)
    assert calls["n"] == 0  # no registry load on construction
    _ = stack._registry
    assert calls["n"] == 1  # loaded on first access
    _ = stack._registry
    assert calls["n"] == 1  # and cached


# --------------------------------------------------------------------------- #
# up() happy path: ordering, URLs, ports, per-instance project naming          #
# --------------------------------------------------------------------------- #


async def test_up_returns_handle_with_urls_and_ports(patched, tmp_path):
    stack = _stack(tmp_path, injection={"travel-injection": "all"})
    handle = await stack.up()
    assert set(handle.server_urls) == {"travel-suite"}
    assert handle.server_urls["travel-suite"].endswith("/mcp")
    assert set(handle.injection_server_urls) == {"travel-injection"}
    assert "TRAVEL_PORT" in handle.ports
    # env brought up before MCP servers spawned
    assert patched["compose_up"], "compose_up was not called"
    assert len(patched["spawned"]) == 2  # one env server + one injection server
    # the per-instance compose project is namespaced by the instance id
    project = patched["compose_up"][0][0]
    assert project.startswith("dtap_") and project.endswith("_travelenv")


def test_project_name_overrides_follow_get_project_name_convention(tmp_path):
    # Mirrors utils.compose_utils.get_project_name: <ENV_UPPER w/ - -> _>_PROJECT_NAME.
    stack = _stack(tmp_path)
    stack._projects = {
        "os-filesystem": "dtap_ab_os-filesystem",
        "terminal": "dtap_ab_terminal",
    }
    assert stack._project_name_overrides() == {
        "OS_FILESYSTEM_PROJECT_NAME": "dtap_ab_os-filesystem",
        "TERMINAL_PROJECT_NAME": "dtap_ab_terminal",
    }


async def test_spawned_server_env_carries_project_name(patched, tmp_path):
    # The exec-based DTAP servers (terminal/research/...) read <ENV>_PROJECT_NAME;
    # every spawned server must see it, set to the env's per-instance compose project.
    stack = _stack(tmp_path)
    await stack.up()
    spawned_env = patched["spawned"][0]["env"]
    assert "TRAVELENV_PROJECT_NAME" in spawned_env
    assert spawned_env["TRAVELENV_PROJECT_NAME"].endswith("_travelenv")


async def test_up_is_idempotent(patched, tmp_path):
    stack = _stack(tmp_path)
    h1 = await stack.up()
    n_after_first = len(patched["compose_up"])
    h2 = await stack.up()
    assert h1 is h2
    assert len(patched["compose_up"]) == n_after_first  # no second bring-up


async def test_up_rejects_gui_server_before_touching_docker(patched, tmp_path):
    def _boom(servers):
        raise ValueError("vision/GUI domain rejected")

    patched["registry"].require_text_only = _boom
    stack = _stack(tmp_path)
    with pytest.raises(ValueError, match="vision/GUI"):
        await stack.up()
    assert patched["compose_up"] == []  # nothing brought up


# --------------------------------------------------------------------------- #
# up() readiness failure -> wrapped with server log tails (the ujson lesson)   #
# --------------------------------------------------------------------------- #


async def test_up_wraps_readiness_failure_with_log_tails(
    patched, tmp_path, monkeypatch
):
    # Make the spawned "server" write a crash to its log, like a missing dep.
    def _spawn_crashing(cmd, *, cwd=None, env=None, log_path=None):
        if log_path:
            Path(log_path).write_text(
                "Traceback...\nModuleNotFoundError: No module named 'ujson'\n"
            )
        return _FakeProc()

    monkeypatch.setattr(lc, "_spawn_process", _spawn_crashing)

    async def _ready_fail(urls, **kw):
        raise RuntimeError("MCP servers failed to become ready: travel-suite")

    monkeypatch.setattr(lc, "_wait_for_ready", _ready_fail)

    stack = _stack(tmp_path)
    with pytest.raises(RuntimeError) as ei:
        await stack.up()
    msg = str(ei.value)
    assert "failed to become ready" in msg
    assert "ujson" in msg  # the actionable crash detail is surfaced, not hidden


def test_server_log_tails_handles_missing_and_present(tmp_path):
    stack = _stack(tmp_path)
    present = tmp_path / "ok.log"
    present.write_text("line1\nline2\nIMPORTANT\n")
    stack._server_logs = {"good": str(present), "gone": str(tmp_path / "missing.log")}
    out = stack._server_log_tails()
    assert "IMPORTANT" in out
    assert "(no log)" in out  # missing file handled, not raised


def test_server_log_tails_empty_when_none_captured(tmp_path):
    stack = _stack(tmp_path)
    assert stack._server_log_tails() == "(no server logs captured)"


# --------------------------------------------------------------------------- #
# reset() and down()                                                           #
# --------------------------------------------------------------------------- #


async def test_reset_resets_each_env(patched, tmp_path):
    stack = _stack(tmp_path)
    await stack.up()
    await stack.reset()
    assert [e for e, _ in patched["reset_env"]] == ["travelenv"]


async def test_down_terminates_procs_and_releases_leases(patched, tmp_path):
    stack = _stack(tmp_path, injection={"travel-injection": "all"})
    await stack.up()
    procs = [*stack._mcp_procs.values(), *stack._inj_procs.values()]
    assert len(procs) == 2
    await stack.down()
    assert all(p.terminated for p in procs)
    assert patched["compose_down"], "compose_down not called"
    assert stack._mcp_procs == {} and stack._inj_procs == {}
    assert stack._leaser.leased == ()  # all port leases released


async def test_down_is_safe_when_compose_down_raises(patched, tmp_path, monkeypatch):
    stack = _stack(tmp_path)
    await stack.up()

    async def _boom(project, cf, *, sudo=None):
        raise RuntimeError("daemon gone")

    monkeypatch.setattr(compose, "compose_down", _boom)
    await stack.down()  # best-effort: must not raise
    assert stack._leaser.leased == ()
