"""Offline tests for the OOB DTAP judge runner.

Every Docker / subprocess / LLM boundary is mocked: ``subprocess.run`` is
monkeypatched to a recorder returning canned JSON, so these tests assert the
ENV wiring (per-instance ports + judge LLM creds + model), the stdin payload
(faithful ``response=final_response``), the stdout->verdict mapping, and the
error paths -- with no real process ever spawned. A real-SDK end-to-end check
lives behind the ``live``/``docker`` markers and is skipped here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from typing import Any

import pytest

from dtap_scaffold import judge_runner as jr

# --------------------------------------------------------------------------- #
# fake subprocess.run                                                          #
# --------------------------------------------------------------------------- #

# Full upstream run_judge() verdict shape (utils/judge_helpers.py:108-116).
UPSTREAM_OK = {
    "task_success": True,
    "task_message": "task done",
    "task_metadata": {"foo": 1},
    "attack_success": False,
    "attack_message": "attack blocked",
    "attack_metadata": {},
    "error": None,
}


class _Recorder:
    """Captures the (cmd, kwargs) of the last subprocess.run call."""

    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        raises: BaseException | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.raises = raises
        self.cmd: list[str] | None = None
        self.kwargs: dict[str, Any] = {}

    def __call__(self, cmd, **kwargs):  # mimics subprocess.run signature we use
        self.cmd = cmd
        self.kwargs = kwargs
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(
            args=cmd, returncode=self.returncode, stdout=self.stdout, stderr=self.stderr
        )

    @property
    def env(self) -> dict[str, str]:
        return self.kwargs["env"]

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.kwargs["input"])


def _install(monkeypatch, rec: _Recorder) -> _Recorder:
    monkeypatch.setattr(jr.subprocess, "run", rec)
    return rec


def _call(**overrides):
    base: dict[str, Any] = dict(
        task_dir="/data/travel/malicious/indirect/x/1",
        final_response="FINAL",
        agent_responses=["t0", "t1", "FINAL"],
        trajectory_json={"task_info": {}, "trajectory": [{"step": 0}]},
        env_ports={"TRAVEL_PORT": 10312, "GMAIL_UI_PORT": 8025},
        judge_model="openai/gpt-4o-2024-05-13",
        judge_api_base="http://proxy.local",
        judge_api_key="sk-judge",
    )
    base.update(overrides)
    return jr.run_dtap_judge(
        base["task_dir"],
        base["final_response"],
        base["agent_responses"],
        base["trajectory_json"],
        base["env_ports"],
        judge_model=base["judge_model"],
        judge_api_base=base["judge_api_base"],
        judge_api_key=base["judge_api_key"],
    )


# --------------------------------------------------------------------------- #
# happy-path mapping                                                           #
# --------------------------------------------------------------------------- #


def test_maps_upstream_verdict_to_frozen_contract(monkeypatch):
    _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    out = _call()
    assert out == {
        "attack_success": False,
        "task_success": True,
        "attack_message": "attack blocked",
        "task_message": "task done",
        "error": None,
    }
    # upstream-only keys are dropped from the frozen contract
    assert "task_metadata" not in out and "attack_metadata" not in out


def test_attack_not_applicable_is_none(monkeypatch):
    verdict = {**UPSTREAM_OK, "attack_success": None, "attack_message": ""}
    _install(monkeypatch, _Recorder(stdout=json.dumps(verdict)))
    out = _call()
    assert out["attack_success"] is None
    assert out["task_success"] is True


def test_none_messages_coerced_to_empty_string(monkeypatch):
    verdict = {
        "task_success": True,
        "attack_success": True,
        "task_message": None,
        "attack_message": None,
        "error": None,
    }
    _install(monkeypatch, _Recorder(stdout=json.dumps(verdict)))
    out = _call()
    assert out["task_message"] == ""
    assert out["attack_message"] == ""


# --------------------------------------------------------------------------- #
# environment wiring                                                           #
# --------------------------------------------------------------------------- #


def test_env_ports_exported_as_strings(monkeypatch):
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call(env_ports={"TRAVEL_PORT": 10312, "CUSTOMER_SERVICE_API_PORT": 8080})
    assert rec.env["TRAVEL_PORT"] == "10312"
    assert rec.env["CUSTOMER_SERVICE_API_PORT"] == "8080"


def test_judge_llm_creds_wired(monkeypatch):
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call(judge_api_base="http://proxy.local", judge_api_key="sk-judge")
    assert rec.env[jr.OPENAI_BASE_URL_ENV] == "http://proxy.local"
    assert rec.env[jr.OPENAI_API_KEY_ENV] == "sk-judge"


def test_judge_model_substitution_explicit(monkeypatch):
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call(judge_model="openai/gpt-4-turbo-2024-04-09")
    assert rec.env[jr.JUDGE_MODEL_ENV] == "openai/gpt-4-turbo-2024-04-09"


def test_judge_model_defaults_when_none(monkeypatch):
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call(judge_model=None)
    assert (
        rec.env[jr.JUDGE_MODEL_ENV]
        == jr.DEFAULT_JUDGE_MODEL
        == "openai/gpt-4o-2024-05-13"
    )


def test_creds_left_to_ambient_when_none(monkeypatch):
    # When the caller passes no creds we must NOT clobber/None them; the ambient
    # env passes through unchanged.
    monkeypatch.delenv(jr.OPENAI_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(jr.OPENAI_API_KEY_ENV, raising=False)
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call(judge_api_base=None, judge_api_key=None)
    assert jr.OPENAI_BASE_URL_ENV not in rec.env
    assert jr.OPENAI_API_KEY_ENV not in rec.env
    # model is still pinned even when creds are ambient
    assert rec.env[jr.JUDGE_MODEL_ENV] == jr.DEFAULT_JUDGE_MODEL


def test_ambient_creds_preserved_when_none(monkeypatch):
    monkeypatch.setenv(jr.OPENAI_BASE_URL_ENV, "http://ambient")
    monkeypatch.setenv(jr.OPENAI_API_KEY_ENV, "sk-ambient")
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call(judge_api_base=None, judge_api_key=None)
    assert rec.env[jr.OPENAI_BASE_URL_ENV] == "http://ambient"
    assert rec.env[jr.OPENAI_API_KEY_ENV] == "sk-ambient"


def test_parent_env_inherited(monkeypatch):
    monkeypatch.setenv("SOME_UNRELATED_VAR", "keepme")
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call()
    assert rec.env["SOME_UNRELATED_VAR"] == "keepme"


def test_empty_env_ports_ok(monkeypatch):
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    out = _call(env_ports={})
    assert out["task_success"] is True
    assert isinstance(rec.env, dict)


# --------------------------------------------------------------------------- #
# stdin payload (the run_judge call arguments)                                 #
# --------------------------------------------------------------------------- #


def test_payload_forwards_final_response_not_agent_list(monkeypatch):
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call(final_response="FINAL", agent_responses=["a", "b", "FINAL"])
    payload = rec.payload
    assert payload["final_response"] == "FINAL"
    assert payload["task_dir"] == "/data/travel/malicious/indirect/x/1"
    # agent_responses is NOT forwarded (faithful: upstream grades final response)
    assert "agent_responses" not in payload


def test_payload_forwards_trajectory(monkeypatch):
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    traj = {"task_info": {"id": 7}, "trajectory": [{"step": 0}, {"step": 1}]}
    _call(trajectory_json=traj)
    assert rec.payload["trajectory_json"] == traj


def test_payload_trajectory_none_ok(monkeypatch):
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    out = _call(trajectory_json=None)
    assert rec.payload["trajectory_json"] is None
    assert out["error"] is None


def test_none_final_response_coerced_to_empty(monkeypatch):
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call(final_response=None)
    assert rec.payload["final_response"] == ""


# --------------------------------------------------------------------------- #
# subprocess invocation shape                                                  #
# --------------------------------------------------------------------------- #


def test_invokes_current_interpreter_with_dash_c(monkeypatch):
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call()
    assert rec.cmd[0] == sys.executable
    assert rec.cmd[1] == "-c"
    assert rec.kwargs["capture_output"] is True
    assert rec.kwargs["text"] is True
    assert rec.kwargs["timeout"] == jr.DEFAULT_JUDGE_TIMEOUT


def test_cwd_is_task_dir_when_it_exists(monkeypatch, tmp_path):
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call(task_dir=str(tmp_path))
    assert rec.kwargs["cwd"] == str(tmp_path)


def test_cwd_none_when_task_dir_missing(monkeypatch):
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call(task_dir="/no/such/dir/anywhere/12345")
    assert rec.kwargs["cwd"] is None


def test_timeout_overridable_via_env(monkeypatch):
    monkeypatch.setenv(jr.JUDGE_TIMEOUT_ENV, "42.5")
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call()
    assert rec.kwargs["timeout"] == 42.5


def test_bad_timeout_env_falls_back(monkeypatch):
    monkeypatch.setenv(jr.JUDGE_TIMEOUT_ENV, "not-a-number")
    rec = _install(monkeypatch, _Recorder(stdout=json.dumps(UPSTREAM_OK)))
    _call()
    assert rec.kwargs["timeout"] == jr.DEFAULT_JUDGE_TIMEOUT


# --------------------------------------------------------------------------- #
# error paths                                                                  #
# --------------------------------------------------------------------------- #


def test_run_judge_internal_error_passthrough(monkeypatch):
    # run_judge succeeded as a process but reported its own error (e.g. no judge.py)
    verdict = {
        "task_success": None,
        "attack_success": None,
        "task_message": "",
        "attack_message": "",
        "error": "judge.py not found in /data/x",
    }
    _install(monkeypatch, _Recorder(stdout=json.dumps(verdict)))
    out = _call()
    assert out["error"] == "judge.py not found in /data/x"
    assert out["attack_success"] is None and out["task_success"] is None


def test_timeout_returns_error_with_none_successes(monkeypatch):
    exc = subprocess.TimeoutExpired(cmd=["python"], timeout=7)
    _install(monkeypatch, _Recorder(raises=exc))
    out = _call()
    assert out["attack_success"] is None and out["task_success"] is None
    assert "timed out" in out["error"]
    assert "7" in out["error"]


def test_launch_failure_returns_error(monkeypatch):
    _install(monkeypatch, _Recorder(raises=OSError("exec format error")))
    out = _call()
    assert out["attack_success"] is None and out["task_success"] is None
    assert "could not be launched" in out["error"]
    assert "OSError" in out["error"]


def test_unparseable_stdout_returns_error(monkeypatch):
    _install(
        monkeypatch, _Recorder(stdout="not json at all", stderr="boom", returncode=1)
    )
    out = _call()
    assert out["attack_success"] is None and out["task_success"] is None
    assert "no parseable JSON" in out["error"]
    assert "returncode=1" in out["error"]
    assert "boom" in out["error"]


def test_empty_stdout_returns_error(monkeypatch):
    _install(monkeypatch, _Recorder(stdout="", stderr="trace", returncode=2))
    out = _call()
    assert out["error"] is not None
    assert out["attack_success"] is None


def test_non_dict_json_stdout_returns_error(monkeypatch):
    _install(monkeypatch, _Recorder(stdout=json.dumps(["a", "list"])))
    out = _call()
    assert out["error"] is not None
    assert out["task_success"] is None


def test_tolerant_parse_ignores_leading_print_noise(monkeypatch):
    noisy = "[JUDGE] loaded trajectory\nsome chatter\n" + json.dumps(UPSTREAM_OK)
    _install(monkeypatch, _Recorder(stdout=noisy))
    out = _call()
    assert out["task_success"] is True
    assert out["attack_success"] is False


# --------------------------------------------------------------------------- #
# child-source faithfulness (static checks, no process spawned)               #
# --------------------------------------------------------------------------- #


@pytest.mark.faithfulness
def test_child_imports_installed_sdk_run_judge():
    assert "from utils.judge_helpers import run_judge" in jr._CHILD_SOURCE


@pytest.mark.faithfulness
def test_child_calls_run_judge_with_faithful_kwargs():
    src = jr._CHILD_SOURCE
    assert "run_judge(" in src
    assert 'response=payload.get("final_response"' in src
    assert 'trajectory=payload.get("trajectory_json")' in src
    # we must NOT override the upstream defaults for these
    assert "check_task" not in src
    assert "check_attack" not in src


@pytest.mark.faithfulness
def test_child_routes_judge_prints_off_stdout():
    # judge print() must go to stderr; the result is written via __stdout__
    assert "sys.stdout = sys.stderr" in jr._CHILD_SOURCE
    assert "sys.__stdout__" in jr._CHILD_SOURCE


# --------------------------------------------------------------------------- #
# real subprocess round-trip against a FAKE utils SDK (offline, no Docker/LLM) #
# --------------------------------------------------------------------------- #

_FAKE_RUN_JUDGE = """
def run_judge(task_dir, response="", *, trajectory=None, check_task=True, check_attack=True):
    # Prove judge stdout chatter does NOT corrupt the result channel.
    print("CHATTER to stdout from the judge")
    import sys
    print("CHATTER to real stderr", file=sys.stderr)
    return {
        "task_success": True,
        "task_message": "ok:" + str(task_dir),
        "task_metadata": {"x": 1},
        "attack_success": (response == "WIN"),
        "attack_message": "resp=" + str(response) + ";traj=" + str(trajectory),
        "attack_metadata": {},
        "error": None,
    }
"""


def _write_fake_sdk(root) -> str:
    pkg = root / "fakesdk" / "utils"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "judge_helpers.py").write_text(_FAKE_RUN_JUDGE, encoding="utf-8")
    return str(root / "fakesdk")


def test_real_subprocess_roundtrip_with_fake_sdk(monkeypatch, tmp_path):
    """Spawn the actual child against a fake ``utils.judge_helpers`` (no real
    process is a server / makes a network or LLM call). Validates: stdin payload
    delivery, ``response=final_response`` forwarding, trajectory forwarding,
    stdout JSON parsing, and that chatty judge ``print()`` is redirected so it
    cannot corrupt the result."""
    fake_root = _write_fake_sdk(tmp_path)
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", fake_root + (os.pathsep + existing if existing else "")
    )
    monkeypatch.setenv(jr.JUDGE_TIMEOUT_ENV, "60")  # never hang the suite

    task_dir = tmp_path / "task"
    task_dir.mkdir()

    out = jr.run_dtap_judge(
        str(task_dir),
        final_response="WIN",
        agent_responses=["ignored", "WIN"],
        trajectory_json={"k": 1},
        env_ports={"DEMO_PORT": 7777},
        judge_model=None,
        judge_api_base="http://proxy",
        judge_api_key="sk-x",
    )
    assert out["error"] is None
    assert out["task_success"] is True
    assert out["attack_success"] is True  # because response == "WIN"
    assert out["task_message"] == "ok:" + str(task_dir)
    assert "resp=WIN" in out["attack_message"]
    assert "traj={'k': 1}" in out["attack_message"]


def test_real_subprocess_child_error_becomes_error_result(monkeypatch, tmp_path):
    """If the child cannot import ``utils`` (no SDK), its own try/except emits an
    error verdict on stdout and the parent maps it -- successes ``None``."""
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "definitely-empty"))
    monkeypatch.setenv(jr.JUDGE_TIMEOUT_ENV, "60")
    # Run from a dir with no `utils` package so the import genuinely fails.
    task_dir = tmp_path / "task2"
    task_dir.mkdir()
    out = jr.run_dtap_judge(
        str(task_dir),
        final_response="x",
        agent_responses=["x"],
        trajectory_json=None,
        env_ports={},
        judge_api_base="http://proxy",
        judge_api_key="sk-x",
    )
    assert out["attack_success"] is None and out["task_success"] is None
    assert out["error"] is not None
    assert "judge subprocess failed" in out["error"]
    assert "ModuleNotFoundError" in out["error"] or "No module named" in out["error"]


# --------------------------------------------------------------------------- #
# live / docker: real-SDK end-to-end (skipped offline)                        #
# --------------------------------------------------------------------------- #

_SDK_PRESENT = importlib.util.find_spec("utils") is not None
_DATASET_ROOT = os.getenv("DTAP_DATASET_ROOT", "")


@pytest.mark.live
@pytest.mark.docker
@pytest.mark.skipif(
    not (_SDK_PRESENT and _DATASET_ROOT and os.path.isdir(_DATASET_ROOT)),
    reason="needs the installed decodingtrust-agent-sdk, a live env (Docker), and DTAP_DATASET_ROOT",
)
def test_real_sdk_judge_smoke():  # pragma: no cover - exercised only where Docker+SDK exist
    # A minimal real-SDK round trip: pick any task dir with a judge.py and run
    # the actual judge against the live env. Requires Docker-up env ports, hence
    # the docker+live markers. Intentionally light: it only asserts the contract
    # shape, since success depends on live state.
    task_dir = None
    for dirpath, _dirs, files in os.walk(_DATASET_ROOT):
        if "judge.py" in files:
            task_dir = dirpath
            break
    if task_dir is None:  # pragma: no cover
        pytest.skip("no judge.py found under DTAP_DATASET_ROOT")
    out = jr.run_dtap_judge(
        task_dir,
        final_response="",
        agent_responses=[""],
        trajectory_json=None,
        env_ports={},
    )
    assert set(out) == {
        "attack_success",
        "task_success",
        "attack_message",
        "task_message",
        "error",
    }
