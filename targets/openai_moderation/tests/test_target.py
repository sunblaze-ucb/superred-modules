"""OpenAIModerationTarget tests: verdict parsing, category extraction, secret
handling, error/retry handling. Offline via httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
from superred.core.controller import TargetFactory
from superred.core.types.events import ControllableInjection, ControllableNoInjection

from openai_moderation_target import (
    INPUT_TAG,
    OpenAIModerationTarget,
    openai_moderation_target_factory,
)

KEY = "sk-SECRET-do-not-leak"


def _transport(captured: list[httpx.Request], response, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if isinstance(response, dict):
            return httpx.Response(status, json=response)
        return httpx.Response(status, text=response)

    return httpx.MockTransport(handler)


def _handlers(text: str | None):
    async def send_event(ev):
        if text is not None:
            return ControllableInjection(event=ev, controllable=ev.controllable, value=text)
        return ControllableNoInjection(event=ev, controllable=ev.controllable)

    return (lambda ev: None), send_event


def _target(transport, **kw) -> OpenAIModerationTarget:
    return OpenAIModerationTarget(api_key=KEY, transport=transport, **kw)


def _result(flagged: bool, categories: dict | None = None) -> dict:
    return {"results": [{"flagged": flagged, "categories": categories or {}}]}


# -- contract ----------------------------------------------------------------
def test_contract() -> None:
    t = _target(_transport([], _result(False)))
    assert {c.name for c in t.config_specs} == {"benign_input"}
    assert {q.name for q in t.query_specs} >= {
        "flagged",
        "flagged_categories",
        "error",
        "sent_input",
    }
    assert t.get_controllables()[0].security_domain == INPUT_TAG


def test_factory_builds_target() -> None:
    fac = openai_moderation_target_factory(api_key=KEY, transport=_transport([], _result(False)))
    assert isinstance(fac, TargetFactory)
    assert isinstance(fac.create(), OpenAIModerationTarget)


# -- verdict -----------------------------------------------------------------
async def test_flagged_true_with_categories() -> None:
    captured: list[httpx.Request] = []
    resp = _result(True, {"violence": True, "hate": False, "self-harm": True})
    t = _target(_transport(captured, resp))
    emit, send = _handlers("something harmful")
    await t.run(emit, send)
    assert t.query("flagged") == "true"
    assert set(json.loads(t.query("flagged_categories"))) == {"violence", "self-harm"}
    # input landed in the request body
    body = json.loads(captured[0].content)
    assert body["input"] == "something harmful" and body["model"] == "omni-moderation-latest"


async def test_flagged_false() -> None:
    t = _target(_transport([], _result(False)))
    emit, send = _handlers("hello there")
    await t.run(emit, send)
    assert t.query("flagged") == "false"
    assert json.loads(t.query("flagged_categories")) == []


async def test_benign_input_when_not_injected() -> None:
    captured: list[httpx.Request] = []
    t = _target(_transport(captured, _result(False)))
    t.set_config("benign_input", "what is the capital of France?")
    emit, send = _handlers(None)
    await t.run(emit, send)
    body = json.loads(captured[0].content)
    assert body["input"] == "what is the capital of France?"


# -- errors / retry ----------------------------------------------------------
async def test_http_error_recorded() -> None:
    t = _target(_transport([], {"error": "bad"}, status=401), max_retries=1)
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("error") == "HTTP 401" and t.query("flagged") == ""


async def test_3xx_redirect_recorded_as_error() -> None:
    # redirects are not followed; a 3xx is not a verdict -> error (claim abstains).
    t = _target(_transport([], "", status=302), max_retries=1)
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("error") == "HTTP 302" and t.query("flagged") == ""


async def test_invalid_json_recorded() -> None:
    t = _target(_transport([], "<html>gateway error</html>", status=200))
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("error") == "invalid JSON response" and t.query("flagged") == ""


async def test_unexpected_shape_recorded() -> None:
    t = _target(_transport([], {"results": []}))
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("flagged") == "" and t.query("error") != ""


async def test_result_without_bool_flagged_is_error_not_empty_verdict() -> None:
    # a parseable 200 whose result lacks a bool `flagged` must set an error, not a
    # contradictory "succeeded, no verdict" state (empty flagged <=> error set).
    t = _target(_transport([], {"results": [{"categories": {"violence": True}}]}))
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("flagged") == "" and t.query("error") != ""


async def test_5xx_retries_then_recovers() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, json={})
        return httpx.Response(200, json=_result(True, {"hate": True}))

    t = _target(httpx.MockTransport(handler), max_retries=3, retry_backoff_base=0.0)
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert calls["n"] == 2 and t.query("flagged") == "true" and t.query("error") == ""


async def test_retry_after_is_clamped(monkeypatch) -> None:  # noqa: ANN001
    import openai_moderation_target.target as mod

    slept: list[float] = []

    async def fake_sleep(d: float) -> None:
        slept.append(d)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "86400"}, json={})

    t = _target(httpx.MockTransport(handler), max_retries=2, max_retry_delay=0.5)
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert slept and max(slept) <= 0.5


# -- secret handling ---------------------------------------------------------
def test_key_never_emitted_in_observables() -> None:
    t = _target(_transport([], _result(False)))
    blob = " ".join(o.content for o in t.get_observables())
    assert "SECRET" not in blob
    assert t.get_observables()[0].content == "omni-moderation-latest"


async def test_key_absent_from_all_queries() -> None:
    t = _target(_transport([], _result(True, {"hate": True})))
    emit, send = _handlers("attack")
    await t.run(emit, send)
    for q in (
        "flagged", "flagged_categories", "raw_response", "http_status", "error", "sent_input"
    ):
        assert "SECRET" not in t.query(q)
    # but the key IS sent on the wire
    captured: list[httpx.Request] = []
    t2 = _target(_transport(captured, _result(False)))
    await t2.run(*_handlers("x"))
    assert captured[0].headers["Authorization"] == f"Bearer {KEY}"


async def test_reset_clears_state() -> None:
    t = _target(_transport([], _result(True, {"hate": True})))
    emit, send = _handlers("x")
    await t.run(emit, send)
    assert t.query("flagged") == "true"
    await t.reset_ephemeral_state()
    assert t.query("flagged") == "" and t.query("flagged_categories") == "[]"
