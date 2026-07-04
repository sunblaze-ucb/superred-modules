"""Offline tests for docker/compose.py parsing + the compose-up command.

No Docker: the subprocess seam (``compose._exec``) is monkeypatched. These pin
the ``ps --format json`` parsing (JSON array AND newline-delimited), the
readiness predicate, and that ``up`` runs plain ``up -d`` (upstream parity, no
``--no-build``).
"""

from __future__ import annotations

import json

from dtap_scaffold.docker import compose


def test_parse_ps_json_array():
    rows = compose._parse_ps_json(json.dumps([{"State": "running"}, {"State": "exited"}]))
    assert [r["State"] for r in rows] == ["running", "exited"]


def test_parse_ps_json_single_object():
    assert compose._parse_ps_json(json.dumps({"State": "running"})) == [{"State": "running"}]


def test_parse_ps_json_ndjson():
    text = '{"State": "running"}\n{"State": "healthy"}\n'
    rows = compose._parse_ps_json(text)
    assert len(rows) == 2 and rows[1]["State"] == "healthy"


def test_parse_ps_json_empty_and_garbage():
    assert compose._parse_ps_json("") == []
    assert compose._parse_ps_json("   ") == []
    # NDJSON with a non-JSON line: the good object survives, garbage is skipped.
    assert compose._parse_ps_json('not-json\n{"State": "running"}') == [{"State": "running"}]


def test_row_ready_running_no_healthcheck():
    assert compose._row_ready({"State": "running", "Health": ""}) is True


def test_row_ready_running_healthy():
    assert compose._row_ready({"State": "running", "Health": "healthy"}) is True


def test_row_ready_running_starting_is_not_ready():
    assert compose._row_ready({"State": "running", "Health": "starting"}) is False


def test_row_ready_not_running():
    assert compose._row_ready({"State": "exited", "Health": "healthy"}) is False


async def test_compose_up_uses_plain_up_d(monkeypatch, tmp_path):
    """Upstream runs plain ``up -d`` (task_executor.py); no ``--no-build``."""
    cmds: list[list[str]] = []

    async def _exec(cmd, *, cwd=None, env=None, timeout=None):
        cmds.append(list(cmd))
        return (0, "", "")

    monkeypatch.setattr(compose, "_exec", _exec)
    cf = tmp_path / "docker-compose.yml"
    cf.write_text("services: {}\n")
    await compose.compose_up("proj", cf, ports={"P": 8080}, sudo=False, pull=False)
    assert len(cmds) == 1
    up = cmds[0]
    assert up[-2:] == ["up", "-d"]
    assert "--no-build" not in up


async def test_compose_ps_returns_rows(monkeypatch, tmp_path):
    async def _exec(cmd, *, cwd=None, env=None, timeout=None):
        return (0, json.dumps([{"State": "running", "Health": "healthy"}]), "")

    monkeypatch.setattr(compose, "_exec", _exec)
    cf = tmp_path / "docker-compose.yml"
    cf.write_text("services: {}\n")
    rows = await compose.compose_ps("proj", cf, sudo=False)
    assert rows == [{"State": "running", "Health": "healthy"}]


async def test_compose_ps_empty_on_nonzero_rc(monkeypatch, tmp_path):
    async def _exec(cmd, *, cwd=None, env=None, timeout=None):
        return (1, "", "err")

    monkeypatch.setattr(compose, "_exec", _exec)
    cf = tmp_path / "docker-compose.yml"
    cf.write_text("services: {}\n")
    assert await compose.compose_ps("proj", cf, sudo=False) == []


async def test_wait_healthy_true_when_all_ready(monkeypatch, tmp_path):
    async def _exec(cmd, *, cwd=None, env=None, timeout=None):
        return (0, json.dumps([{"State": "running", "Health": "healthy"}]), "")

    monkeypatch.setattr(compose, "_exec", _exec)
    cf = tmp_path / "docker-compose.yml"
    cf.write_text("services: {}\n")
    assert await compose.wait_healthy("proj", cf, sudo=False, timeout=1, interval=0.01) is True
