"""Offline tests for ``DockerEnvStack`` (the per-instance Docker + MCP lifecycle).

No Docker daemon and no MCP servers: the collaborator seams
(``compose.*`` / ``reset.*`` / ``env_registry.load`` / ``_spawn_process`` /
``_wait_for_ready``) are monkeypatched, and the real ``PortLeaser`` /
``make_instance_state`` run against ``tmp_path`` (the leaser still bind-tests
loopback sockets and writes its lock files under a per-test ``tmp_path`` lock
dir, so no shared state leaks between tests). These tests pin the lifecycle's
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

    def reset_script_timeout(self, env):
        return 180  # distinctive per-env value (mimics env.yaml terminal=180)

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
        return {"environments": {"travelenv": {"reset_scripts": {"default": "reset.sh"}}}}


class _FakeProc:
    def __init__(self) -> None:
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Install all collaborator fakes; return a recorder of side effects."""
    monkeypatch.setenv("DT_PORT_LOCK_DIR", str(tmp_path / "port_locks"))
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
        rec["setup"].append({"cmd": list(cmd), "cwd": cwd, "env": dict(env or {})})
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
    env = lc._server_env(cfg, "TRAVEL_MCP_PORT", 12345, {"TRAVEL_PORT": 8080}, {"X": "y"})
    assert env["PATH"] == os.environ["PATH"]  # inherited -> python3 is findable
    assert env["DTAP_SENTINEL_VAR"] == "keepme"
    assert env["TRAVEL_MCP_PORT"] == "12345"  # own listen port
    assert env["API"] == "http://h:8080"  # rendered from container_ports
    assert env["X"] == "y"  # extra overrides win


def test_server_env_task_overrides_applied_last(monkeypatch):
    """Regression (gap #2): the per-task config.yaml env_vars (USER_ACCESS_TOKEN +
    per-task creds) are upstream's HIGHEST-priority tier -- they must land in the MCP
    server env AND win over every lower tier (os.environ, mcp.yaml env, ports, the
    state/project extra). Before the fix they were dropped entirely."""
    monkeypatch.setenv("USER_ACCESS_TOKEN", "ambient-empty")
    cfg = {"env": {"USER_ACCESS_TOKEN": "yaml-default", "API": "http://h:${TRAVEL_PORT}"}}
    overrides = {"USER_ACCESS_TOKEN": "alice-token", "FINANCE_ACCOUNTS_JSON": '{"a": 1}'}
    # the state/project `extra` tier ALSO sets USER_ACCESS_TOKEN, so this pins the
    # override-vs-extra boundary too (a mutation applying extra last would survive
    # without a shared key across those two top tiers).
    extra = {"X": "y", "USER_ACCESS_TOKEN": "extra-loses"}
    env = lc._server_env(cfg, "TRAVEL_MCP_PORT", 12345, {"TRAVEL_PORT": 8080}, extra, overrides)
    # the per-task identity wins over EVERY lower tier: os.environ, mcp.yaml env, extra
    assert env["USER_ACCESS_TOKEN"] == "alice-token"
    # per-task credential lands (was absent entirely before the fix)
    assert env["FINANCE_ACCOUNTS_JSON"] == '{"a": 1}'
    # lower tiers still present + unshadowed
    assert env["X"] == "y"
    assert env["API"] == "http://h:8080"


def test_server_env_empty_task_override_does_not_shadow_lower_tier():
    """The apply-side truthy filter, pinned independently of the parse-side one: an
    EMPTY task override must NOT overwrite a lower tier (upstream mcp_helpers applies
    only `if env_val`). Deleting the `if value:` guard in _server_env fails this."""
    env = lc._server_env({"env": {}}, "P", 1, {}, {"K": "kept"}, {"K": ""})
    assert env["K"] == "kept"  # the empty override did not shadow the state/extra tier


def test_server_env_no_task_overrides_is_unchanged(monkeypatch):
    """Omitting task_overrides leaves the env exactly as the lower tiers built it."""
    cfg = {"env": {"K": "v"}}
    env = lc._server_env(cfg, "P", 1, {}, {"X": "y"})
    assert env["K"] == "v" and env["X"] == "y"


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


async def test_setup_env_carries_project_name(patched, tmp_path):
    # setup.sh seeds env state by exec-ing into the container via <ENV>_PROJECT_NAME;
    # the seed silently no-ops if it's absent, so the setup exec env must carry it.
    task = tmp_path / "task"
    task.mkdir()
    (task / "setup.sh").write_text("#!/bin/sh\necho seed\n")
    stack = lc.DockerEnvStack(
        active_servers=("travel-suite",),
        injection_config=None,
        task_dir=str(task),
        state_root=str(tmp_path / "state"),
    )
    await stack.up()
    setup_calls = [c for c in patched["setup"] if c["cmd"][:1] == ["bash"]]
    assert setup_calls, "setup.sh was not executed"
    assert "TRAVELENV_PROJECT_NAME" in setup_calls[0]["env"]


async def test_setup_failure_surfaces_stdout_when_stderr_empty(patched, tmp_path, monkeypatch):
    # A seeder that fails with its message on STDOUT and an EMPTY stderr (e.g. crm's
    # `curl -s` reset against a not-yet-ready service) must still produce a
    # diagnosable error, not an opaque "setup.sh failed: ".
    task = tmp_path / "task"
    task.mkdir()
    (task / "setup.sh").write_text("#!/bin/sh\necho boom\n")

    async def _exec_fail(cmd, *, cwd=None, env=None, timeout=None):
        return (1, "curl: (7) Failed to connect to slack API", "")  # stdout has it, stderr empty

    monkeypatch.setattr(compose, "_exec", _exec_fail)
    stack = lc.DockerEnvStack(
        active_servers=("travel-suite",),
        injection_config=None,
        task_dir=str(task),
        state_root=str(tmp_path / "state"),
    )
    with pytest.raises(RuntimeError, match="Failed to connect to slack API"):
        await stack.up()


async def test_spawned_server_env_carries_project_name(patched, tmp_path):
    # The exec-based DTAP servers (terminal/research/...) read <ENV>_PROJECT_NAME;
    # every spawned server must see it, set to the env's per-instance compose project.
    stack = _stack(tmp_path)
    await stack.up()
    spawned_env = patched["spawned"][0]["env"]
    assert "TRAVELENV_PROJECT_NAME" in spawned_env
    assert spawned_env["TRAVELENV_PROJECT_NAME"].endswith("_travelenv")


async def test_up_handle_exposes_project_names(patched, tmp_path):
    """Regression (gap #1, stack level): EnvHandle.project_names is populated so the
    target can surface it (ENV_PROJECT_NAMES) to the OOB judge subprocess."""
    handle = await _stack(tmp_path).up()
    assert "TRAVELENV_PROJECT_NAME" in handle.project_names
    assert handle.project_names["TRAVELENV_PROJECT_NAME"].endswith("_travelenv")


async def test_spawned_server_env_carries_per_task_env_vars(patched, tmp_path):
    """Regression (gap #2): per-task, per-server env_vars reach the spawned MCP server
    process (keyed by server name), so its acting identity/creds match the seeded
    state the judge verifies. Before the fix they were dropped entirely."""
    stack = lc.DockerEnvStack(
        active_servers=("travel-suite",),
        injection_config=None,
        task_dir=None,
        state_root=str(tmp_path / "state"),
        server_env_overrides={"travel-suite": {"USER_ACCESS_TOKEN": "alice-token"}},
    )
    await stack.up()
    spawned_env = patched["spawned"][0]["env"]
    assert spawned_env["USER_ACCESS_TOKEN"] == "alice-token"


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


async def test_up_wraps_readiness_failure_with_log_tails(patched, tmp_path, monkeypatch):
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


async def test_reset_threads_env_script_timeout(patched, tmp_path, monkeypatch):
    # The env's reset_script_timeout (env.yaml; terminal=180) must reach
    # reset_environment, not be silently dropped to the default. Upstream
    # task_executor._reset_instance passes env_def.get("reset_script_timeout", 60).
    calls: list[dict] = []

    async def _capture(env_name, ports, env_config, **kw):
        calls.append({"env": env_name, **kw})

    monkeypatch.setattr(reset_mod, "reset_environment", _capture)
    stack = _stack(tmp_path)
    await stack.up()
    await stack.reset()
    assert calls and calls[0]["script_timeout"] == 180


async def test_down_terminates_procs_and_releases_leases(patched, tmp_path):
    stack = _stack(tmp_path, injection={"travel-injection": "all"})
    await stack.up()
    procs = [*stack._mcp_procs.values(), *stack._inj_procs.values()]
    assert len(procs) == 2
    # up() mkdir'd the per-instance host state + log dirs; capture them before down().
    state_dir = stack._state.state_dir
    logs_dir = Path(stack._state_root) / f"dtap_logs_{stack._iid}"
    assert state_dir.is_dir() and logs_dir.is_dir()
    await stack.down()
    assert all(p.terminated for p in procs)
    assert patched["compose_down"], "compose_down not called"
    assert stack._mcp_procs == {} and stack._inj_procs == {}
    assert stack._leaser.leased == ()  # all port leases released
    # down() reclaims the host state + log dirs (no per-task leak)
    assert not state_dir.exists() and not logs_dir.exists()


def test_port_leaser_reclaims_dead_pid_lock(tmp_path):
    """A lock file left by a crashed instance (dead PID) must be reclaimable, or the
    port is orphaned forever. _try_claim reads the PID, sees it is not alive, unlinks
    the stale lock, and re-claims."""
    leaser = lc.ports_mod.PortLeaser(port_range=(21000, 21000), lock_dir=str(tmp_path))
    # forge a stale lock for the only port, owned by a definitely-dead PID
    stale = leaser._lock_path(21000)
    stale.write_text("2147483646")  # a PID that does not exist
    port = leaser.lease("x")  # must reclaim the stale lock, not raise "no free port"
    assert port == 21000
    assert leaser.leased == (21000,)
    leaser.release_all()


def test_port_leaser_respects_live_pid_lock(tmp_path):
    """A lock owned by a LIVE process (our own PID) must be respected -- never stolen;
    the leaser skips that port (and here, with a 1-port range, fails to lease)."""
    import os as _os

    leaser = lc.ports_mod.PortLeaser(port_range=(21001, 21001), lock_dir=str(tmp_path))
    leaser._lock_path(21001).write_text(str(_os.getpid()))  # a live PID (this test proc)
    with pytest.raises(RuntimeError, match="unable to lease"):
        leaser.lease("x")  # the only port is held by a live pid -> not reclaimed


def test_port_leaser_leaves_malformed_lock_intact(tmp_path):
    """A lock with an unreadable/malformed PID is conservatively left intact (never
    stolen), so a corrupt lock cannot cause a live instance's port to be reused."""
    leaser = lc.ports_mod.PortLeaser(port_range=(21002, 21002), lock_dir=str(tmp_path))
    leaser._lock_path(21002).write_text("not-a-pid")
    with pytest.raises(RuntimeError, match="unable to lease"):
        leaser.lease("x")


async def test_down_is_safe_when_compose_down_raises(patched, tmp_path, monkeypatch):
    stack = _stack(tmp_path)
    await stack.up()

    async def _boom(project, cf, *, sudo=None):
        raise RuntimeError("daemon gone")

    monkeypatch.setattr(compose, "compose_down", _boom)
    await stack.down()  # best-effort: must not raise
    assert stack._leaser.leased == ()


async def test_reset_reseeds_and_forwards_project_and_compose(patched, tmp_path, monkeypatch):
    # reset() must (1) re-run setup.sh to re-seed the env, and (2) forward the
    # env's per-instance project_name + compose_file to reset_environment (needed
    # for the docker-exec script fallback).
    task = tmp_path / "task"
    task.mkdir()
    (task / "setup.sh").write_text("#!/bin/sh\necho seed\n")

    calls: list[dict] = []

    async def _capture(env_name, ports, env_config, **kw):
        calls.append({"env": env_name, **kw})

    monkeypatch.setattr(reset_mod, "reset_environment", _capture)

    stack = lc.DockerEnvStack(
        active_servers=("travel-suite",),
        injection_config=None,
        task_dir=str(task),
        state_root=str(tmp_path / "state"),
    )
    await stack.up()
    setup_after_up = sum(1 for c in patched["setup"] if c["cmd"][:1] == ["bash"])
    await stack.reset()
    setup_after_reset = sum(1 for c in patched["setup"] if c["cmd"][:1] == ["bash"])

    assert setup_after_up == 1  # seeded once during up()
    assert setup_after_reset == 2  # re-seeded again during reset()
    assert len(calls) == 1
    assert calls[0]["env"] == "travelenv"
    assert calls[0]["project_name"].endswith("_travelenv")
    assert str(calls[0]["compose_file"]).endswith("docker-compose.yml")
