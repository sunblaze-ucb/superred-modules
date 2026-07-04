"""Unit tests for reviewer-fix areas: auth on host servers, reset RPC, proxy injection."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import web

from openclaw_target.injection_server import InjectionServer
from openclaw_target.proxy_llm import LLMProxy
from openclaw_target.target import MODEL_RESPONSE_CTRL, OpenClawTarget
from openclaw_target.ws_client import OpenClawWSClient

from superred.core.types.events import ControllableInjection, ObservableEvent


# -- injection server auth ----------------------------------------------------


@pytest.mark.asyncio
async def test_injection_server_rejects_missing_token() -> None:
    server = InjectionServer(
        handler=lambda *_: None,
        host="127.0.0.1",
        port=0,
        auth_token="secret-callback",
    )
    await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{server.actual_port}/hook",
                json={"hook": "before_tool_call", "toolName": "read"},
            ) as resp:
                assert resp.status == 401
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_injection_server_accepts_bearer_token() -> None:
    server = InjectionServer(
        handler=lambda *_: {"toolResult": "injected"},
        host="127.0.0.1",
        port=0,
        auth_token="secret-callback",
    )
    await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{server.actual_port}/hook",
                json={"hook": "before_tool_call", "toolName": "read"},
                headers={"Authorization": "Bearer secret-callback"},
            ) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert body == {"toolResult": "injected"}
    finally:
        await server.stop()


# -- LLM proxy auth + response injection --------------------------------------


@pytest.mark.asyncio
async def test_llm_proxy_rejects_missing_token() -> None:
    proxy = LLMProxy(
        upstream_base_url="http://example.invalid",
        upstream_api_key="sk-upstream",
        host="127.0.0.1",
        port=0,
        inbound_token="proxy-secret",
    )
    await proxy.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{proxy.actual_port}/v1/chat/completions",
                json={"model": "gpt-5", "messages": []},
            ) as resp:
                assert resp.status == 401
    finally:
        await proxy.stop()


@pytest.mark.asyncio
async def test_llm_proxy_injects_response_text() -> None:
    proxy = LLMProxy(
        upstream_base_url="http://example.invalid",
        upstream_api_key="sk-upstream",
        host="127.0.0.1",
        port=0,
    )
    resp_body = {
        "choices": [{"message": {"role": "assistant", "content": "original"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    proxy.response_injection = "INJECTED"
    proxy._inject_response(resp_body)
    assert resp_body["choices"][0]["message"]["content"] == "original\nINJECTED"


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("https://api.openai.com", "https://api.openai.com/v1/chat/completions"),
        ("https://api.openai.com/", "https://api.openai.com/v1/chat/completions"),
        (
            # Gemini's OpenAI-compatible endpoint already ends in "/v1" -
            # verified live: naively always appending "/v1/chat/completions"
            # here produces ".../v1/v1/chat/completions", which 404s.
            "https://generativelanguage.googleapis.com/v1beta/openai/v1",
            "https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions",
        ),
    ],
)
def test_llm_proxy_upstream_url_does_not_double_append_v1(base: str, expected: str) -> None:
    proxy = LLMProxy(upstream_base_url=base, upstream_api_key="k")
    assert proxy._upstream_chat_completions_url() == expected


@asynccontextmanager
async def _real_streaming_upstream_stub(
    *, reply: str = "Real upstream reply.",
) -> AsyncIterator[tuple[str, list[bool | None]]]:
    """A stub upstream that behaves like a *real* provider, not our other
    test stubs: if the request says ``stream: true`` it replies with a
    genuine SSE body (``text/event-stream``), never a plain JSON object.

    Yields ``(base_url, received_stream_flags)`` - the list records the
    ``stream`` field of every request this stub receives, so callers can
    assert the proxy never forwards ``stream: true`` upstream.
    """
    received_stream_flags: list[bool | None] = []

    async def completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        received_stream_flags.append(body.get("stream"))
        if body.get("stream"):
            # A real provider actually streams SSE for stream:true - never
            # a single JSON blob. If the proxy ever regresses to forwarding
            # stream:true upstream, this reproduces the live Gemini crash
            # (resp.json() raising ContentTypeError on an SSE body).
            resp = web.StreamResponse(
                headers={"Content-Type": "text/event-stream"},
            )
            await resp.prepare(request)
            await resp.write(
                b'data: {"choices":[{"delta":{"content":"should not be used"}}]}\n\n',
            )
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        return web.json_response(
            {
                "choices": [{"message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}", received_stream_flags
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_llm_proxy_forces_non_streaming_upstream_for_real_providers() -> None:
    """Regression test for a live bug (reproduced against Gemini via a real
    API key): OpenClaw's real client always sends ``stream: true``
    (verified against ``openclaw/openclaw
    src/llm/providers/openai-completions.ts``); a *real* provider honors it
    and replies with SSE, and ``LLMProxy`` used to forward ``stream: true``
    upstream unchanged and then call ``resp.json()`` on that SSE body,
    raising ``ContentTypeError`` and crashing every real-model run. The
    proxy must always request the non-streaming shape from upstream
    (OpenClaw's client already tolerates a plain JSON reply to its
    ``stream: true`` request - every stub-LLM test in this suite proves
    that), regardless of what the inbound request asked for.
    """
    async with _real_streaming_upstream_stub() as (base_url, received_stream_flags):
        proxy = LLMProxy(
            upstream_base_url=base_url,
            upstream_api_key="sk-upstream",
            host="127.0.0.1",
            port=0,
        )
        proxy.response_injection = "PROXY-INJECTED-TEXT"
        await proxy.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{proxy.actual_port}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        # OpenClaw's real client always sets this.
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                ) as resp:
                    assert resp.status == 200
                    assert resp.content_type == "application/json"
                    body = await resp.json()
        finally:
            await proxy.stop()

    assert received_stream_flags == [False], (
        f"proxy forwarded stream={received_stream_flags} upstream; "
        "must always force stream=false against the real provider"
    )
    assert body["choices"][0]["message"]["content"] == (
        "Real upstream reply.\nPROXY-INJECTED-TEXT"
    )
    assert len(proxy.records) == 1
    assert proxy.records[0].response_text == "Real upstream reply.\nPROXY-INJECTED-TEXT"


@pytest.mark.asyncio
async def test_llm_proxy_tolerates_non_json_upstream_error_body() -> None:
    """Some providers return plain-text/HTML error bodies (e.g. a 404 page
    from a misconfigured base URL, reproduced live). ``resp.json()`` on that
    used to raise an uncaught ``ContentTypeError``; the proxy must degrade
    to a structured error instead of crashing the handler."""

    async def not_found(request: web.Request) -> web.Response:
        return web.Response(status=404, content_type="text/html", text="<html>nope</html>")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", not_found)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    try:
        proxy = LLMProxy(
            upstream_base_url=f"http://127.0.0.1:{port}",
            upstream_api_key="sk-upstream",
            host="127.0.0.1",
            port=0,
        )
        await proxy.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{proxy.actual_port}/v1/chat/completions",
                    json={"model": "m", "messages": [], "stream": True},
                ) as resp:
                    assert resp.status == 404
                    body = await resp.json()
                    assert "error" in body
        finally:
            await proxy.stop()
    finally:
        await runner.cleanup()


# -- sessions.reset RPC ---------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_session_uses_key_param() -> None:
    client = OpenClawWSClient(gateway_url="ws://127.0.0.1:0", auth_token="t")
    client.rpc = AsyncMock(return_value={})  # type: ignore[method-assign]

    await client.reset_session("my-session")

    client.rpc.assert_awaited_once_with("sessions.reset", {"key": "my-session"})


@pytest.mark.asyncio
async def test_reset_session_raises_on_rpc_error() -> None:
    client = OpenClawWSClient(gateway_url="ws://127.0.0.1:0", auth_token="t")
    client.rpc = AsyncMock(return_value={"error": "invalid params"})  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="sessions.reset failed"):
        await client.reset_session("my-session")


# -- tool catalogue emission + model-response controllable ----------------------


@pytest.mark.asyncio
async def test_run_emits_tool_catalogue_observable() -> None:
    target = OpenClawTarget(auth_token="t", gateway_url="ws://127.0.0.1:0")
    target._cached_tool_catalog = json.dumps({"tools": [{"name": "exec"}]})

    fake_client = AsyncMock()
    fake_client.run_agent = AsyncMock(
        return_value=MagicMock(
            assistant_text="ok",
            tool_calls=[],
            events=[],
            error=None,
        ),
    )
    target._client = fake_client

    emitted: list[ObservableEvent] = []

    def emit(evt: ObservableEvent) -> None:
        emitted.append(evt)

    async def send_event(event: object) -> ControllableInjection:
        return ControllableInjection(
            event=event,  # type: ignore[arg-type]
            controllable=getattr(event, "controllable"),
            value="hello",
        )

    with patch.object(target, "_ensure_connected", AsyncMock(return_value=fake_client)):
        await target.run(emit, send_event)

    tool_events = [e for e in emitted if e.observable.name == "tool_list"]
    assert len(tool_events) == 1
    assert "exec" in tool_events[0].content


def test_llm_proxy_controllables_include_model_response() -> None:
    target = OpenClawTarget(enable_llm_proxy=True, provider_base_url="http://x")
    names = {c.name for c in target.get_controllables()}
    assert "model_response_injection" in names
    assert MODEL_RESPONSE_CTRL.name == "model_response_injection"


def test_factory_enables_proxy_when_provider_set() -> None:
    from openclaw_target.factory import openclaw_target_factory

    factory = openclaw_target_factory(
        provider_base_url="http://127.0.0.1:9000",
        provider_api_key="sk-test",
    )
    target = factory.create()
    assert target._enable_llm_proxy is True


@pytest.mark.asyncio
async def test_warmup_populates_tool_catalog_for_get_observables() -> None:
    target = OpenClawTarget(auth_token="t", gateway_url="ws://127.0.0.1:0")
    assert target.get_observables()[2].content == ""

    fake_client = AsyncMock()

    async def fake_ensure() -> AsyncMock:
        target._cached_tool_catalog = json.dumps(
            {"tools": [{"name": "exec"}]}, indent=2,
        )
        target._client = fake_client
        return fake_client

    with patch.object(target, "_ensure_connected", side_effect=fake_ensure):
        await target.warmup_static_observables()

    catalog_obs = next(
        o for o in target.get_observables() if o.observable.name == "tool_list"
    )
    assert "exec" in catalog_obs.content
