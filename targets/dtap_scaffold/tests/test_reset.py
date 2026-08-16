"""Offline tests for docker/reset.py (endpoints-then-scripts reset).

No Docker, no network: the HTTP seam (``reset._http_post``) and the compose-exec
seam (``compose._exec``) are monkeypatched. These pin the fallback ordering, the
no-op branch, the unresolved-placeholder skip, and the retry/2xx handling.
"""

from __future__ import annotations

import urllib.error

import pytest

from dtap_scaffold.docker import compose, reset


def _cfg(env_name="travel", *, endpoints=None, scripts=None):
    env_def: dict = {}
    if endpoints is not None:
        env_def["reset_endpoints"] = endpoints
    if scripts is not None:
        env_def["reset_scripts"] = scripts
    return {"environments": {env_name: env_def}}


def test_render_template_resolves_braced_vars():
    assert reset.render_template("http://h:${PORT}/reset", {"PORT": 10300}) == (
        "http://h:10300/reset"
    )


async def test_reset_environment_noop_when_nothing_configured(monkeypatch):
    called = {"http": 0, "exec": 0}

    def _http(*a, **k):
        called["http"] += 1
        return 200

    async def _exec(*a, **k):
        called["exec"] += 1
        return (0, "", "")

    monkeypatch.setattr(reset, "_http_post", _http)
    monkeypatch.setattr(compose, "_exec", _exec)
    await reset.reset_environment("travel", {}, _cfg("travel"))
    assert called == {"http": 0, "exec": 0}  # nothing configured -> silent no-op


async def test_reset_environment_calls_endpoint_with_resolved_url(monkeypatch):
    seen: list[str] = []

    def _http(url, *, method="POST", timeout=30):
        seen.append(url)
        return 200

    monkeypatch.setattr(reset, "_http_post", _http)
    cfg = _cfg(
        "travel",
        endpoints={"api": {"url": "http://127.0.0.1:${TRAVEL_PORT}/reset"}},
    )
    await reset.reset_environment("travel", {"TRAVEL_PORT": 10300}, cfg)
    assert seen == ["http://127.0.0.1:10300/reset"]  # ${VAR} resolved from ports


async def test_reset_via_endpoints_skips_unresolved_placeholder(monkeypatch):
    calls = {"n": 0}

    def _http(url, *, method="POST", timeout=30):
        calls["n"] += 1
        return 200

    monkeypatch.setattr(reset, "_http_post", _http)
    cfg = _cfg("travel", endpoints={"api": {"url": "http://h:${MISSING}/reset"}})
    # ${MISSING} not in ports -> URL still holds "${" -> endpoint skipped, no raise.
    await reset.reset_via_endpoints("travel", {"TRAVEL_PORT": 10300}, cfg)
    assert calls["n"] == 0


async def test_reset_via_endpoints_retries_then_succeeds(monkeypatch):
    statuses = iter([500, 200])

    def _http(url, *, method="POST", timeout=30):
        return next(statuses)

    monkeypatch.setattr(reset, "_http_post", _http)
    cfg = _cfg("travel", endpoints={"api": {"url": "http://h:8080/reset"}})
    # 500 then 200 -> succeeds without raising (retry/2xx).
    await reset.reset_via_endpoints("travel", {}, cfg, max_retries=3, retry_delay=0)


async def test_reset_environment_endpoint_fails_no_scripts_raises(monkeypatch):
    def _http(url, *, method="POST", timeout=30):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(reset, "_http_post", _http)
    cfg = _cfg("travel", endpoints={"api": {"url": "http://h:8080/reset"}})
    with pytest.raises(RuntimeError):
        await reset.reset_environment("travel", {}, cfg, max_retries=1)


async def test_reset_environment_falls_back_to_scripts(monkeypatch, tmp_path):
    def _http(url, *, method="POST", timeout=30):
        return 500  # endpoint keeps failing

    execs: list[list[str]] = []

    async def _exec(cmd, *, cwd=None, env=None, timeout=None):
        execs.append(list(cmd))
        return (0, "", "")

    monkeypatch.setattr(reset, "_http_post", _http)
    monkeypatch.setattr(compose, "_exec", _exec)
    cfg = _cfg(
        "travel",
        endpoints={"api": {"url": "http://h:8080/reset"}},
        scripts={"travel-api": "/app/reset.sh"},
    )
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n")
    await reset.reset_environment(
        "travel",
        {},
        cfg,
        project_name="dtap_x_travel",
        compose_file=compose_file,
        sudo=False,
        max_retries=1,
    )
    # endpoint failed -> script fallback ran a `compose exec -T <service> ...`.
    assert execs, "script fallback did not run"
    assert "exec" in execs[0] and "travel-api" in execs[0]


async def test_reset_via_scripts_raises_contextual_error_on_nonzero_rc(monkeypatch, tmp_path):
    async def _exec(cmd, *, cwd=None, env=None, timeout=None):
        return (1, "", "boom")

    monkeypatch.setattr(compose, "_exec", _exec)
    cfg = _cfg("travel", scripts={"travel-api": "/app/reset.sh"})
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n")
    with pytest.raises(reset.ResetScriptError, match="exit code 1") as raised:
        await reset.reset_via_scripts("travel", "dtap_x_travel", compose_file, cfg, sudo=False)
    message = str(raised.value)
    assert "travel/travel-api" in message
    assert "dtap_x_travel" in message
    assert "/app/reset.sh" in message
    assert "boom" in message


async def test_reset_via_scripts_wraps_timeout_with_context(monkeypatch, tmp_path):
    async def _exec(cmd, *, cwd=None, env=None, timeout=None):
        raise TimeoutError

    monkeypatch.setattr(compose, "_exec", _exec)
    cfg = _cfg("research", scripts={"research-api": "/reset.sh"})
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n")
    with pytest.raises(reset.ResetScriptError, match="timed out") as raised:
        await reset.reset_via_scripts(
            "research",
            "dtap_x_research",
            compose_file,
            cfg,
            sudo=False,
            timeout=7,
        )
    message = str(raised.value)
    assert "research/research-api" in message
    assert "after 7s" in message
    assert "dtap_x_research" in message
    assert "/reset.sh" in message
    assert isinstance(raised.value.__cause__, TimeoutError)
