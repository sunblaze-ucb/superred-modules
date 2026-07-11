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
from openclaw_target.target import MODEL_RESPONSE_CTRL, MODEL_SYSTEM_PROMPT_CTRL, OpenClawTarget
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


def test_llm_proxy_injects_system_prompt_text() -> None:
    proxy = LLMProxy(
        upstream_base_url="http://example.invalid",
        upstream_api_key="sk-upstream",
        host="127.0.0.1",
        port=0,
    )
    messages = [{"role": "system", "content": "Base system."}]
    proxy.system_prompt_injection = "APPENDED-SYSTEM-TEXT"
    proxy._inject_system_prompt(messages)
    assert messages[0]["content"] == "Base system.\nAPPENDED-SYSTEM-TEXT"


def test_llm_proxy_injects_system_prompt_when_missing() -> None:
    proxy = LLMProxy(
        upstream_base_url="http://example.invalid",
        upstream_api_key="sk-upstream",
        host="127.0.0.1",
        port=0,
    )
    messages = [{"role": "user", "content": "hi"}]
    proxy.system_prompt_injection = "NEW-SYSTEM-TEXT"
    proxy._inject_system_prompt(messages)
    assert messages[0] == {"role": "system", "content": "NEW-SYSTEM-TEXT"}
    assert messages[1] == {"role": "user", "content": "hi"}


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


@pytest.mark.parametrize(
    ("qualified", "expected"),
    [
        ("google/gemini-2.5-flash", "gemini-2.5-flash"),
        ("openai/gpt-4o-mini", "gpt-4o-mini"),
        ("gemini-2.5-flash", "gemini-2.5-flash"),
    ],
)
def test_llm_proxy_normalizes_qualified_upstream_model(
    qualified: str, expected: str,
) -> None:
    assert LLMProxy._normalize_upstream_model(qualified) == expected


def test_llm_proxy_coerces_list_shaped_upstream_error() -> None:
    payload = [{"error": {"code": 400, "message": "bad model", "status": "INVALID_ARGUMENT"}}]
    coerced = LLMProxy._coerce_completion_response(payload, status=400)
    assert coerced == {"error": payload[0]["error"]}


@pytest.mark.asyncio
async def test_llm_proxy_forwards_bare_model_name_upstream() -> None:
    """Regression: OpenClaw sends ``google/<model>``; upstream must receive
    the bare model id (verified live against Gemini)."""
    received_models: list[str] = []

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        received_models.append(body.get("model", ""))
        return web.json_response(
            {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
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
                    json={
                        "model": "google/gemini-2.5-flash",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                ) as resp:
                    assert resp.status == 200
        finally:
            await proxy.stop()
    finally:
        await runner.cleanup()

    assert received_models == ["gemini-2.5-flash"]


_REAL_SSE_CHUNKS = [
    {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
     "model": "test-model", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
    {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
     "model": "test-model", "choices": [{"index": 0, "delta": {"content": "Real upstream "}, "finish_reason": None}]},
    {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
     "model": "test-model", "choices": [{"index": 0, "delta": {"content": "reply."}, "finish_reason": None}]},
    {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
     "model": "test-model", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
     "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
]


@asynccontextmanager
async def _real_streaming_upstream_stub(
    *, chunks: list[dict] | None = None,
) -> AsyncIterator[tuple[str, list[bool | None]]]:
    """A stub upstream that behaves like a *real* provider, not our other
    test stubs: if the request says ``stream: true`` it replies with a
    genuine multi-chunk SSE body (``text/event-stream``) delivered
    incrementally (one real network write per chunk, like an actual model
    generating token by token), never a single JSON blob.

    Yields ``(base_url, received_stream_flags)`` - the list records the
    ``stream`` field of every request this stub receives.
    """
    received_stream_flags: list[bool | None] = []

    async def completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        received_stream_flags.append(body.get("stream"))
        if body.get("stream"):
            resp = web.StreamResponse(
                headers={"Content-Type": "text/event-stream"},
            )
            await resp.prepare(request)
            for chunk in (chunks if chunks is not None else _REAL_SSE_CHUNKS):
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        return web.json_response(
            {
                "choices": [{"message": {"role": "assistant", "content": "Real upstream reply."}}],
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


def _parse_sse_body(raw: bytes) -> list[dict]:
    frames = []
    for line in raw.decode().splitlines():
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            continue
        frames.append(json.loads(data))
    return frames


@pytest.mark.asyncio
async def test_llm_proxy_relays_real_provider_sse_stream_live() -> None:
    """Regression test for a live bug (reproduced against Gemini via a real
    API key): OpenClaw's real client always sends ``stream: true``
    (verified against ``openclaw/openclaw
    src/llm/providers/openai-completions.ts``); a *real* provider honors it
    and replies with SSE, and ``LLMProxy`` used to call ``resp.json()`` on
    that SSE body, raising ``ContentTypeError`` and crashing every
    real-model run.

    The fix must preserve *genuine incremental* delivery end to end (an
    earlier fix forced ``stream: false`` upstream, which "worked" but
    silently collapsed live token-by-token deltas into a single chunk
    delivered only once generation fully finished - verified live against
    Gemini: 3 incremental deltas direct vs. 1 batched delta through that
    version). This test asserts the proxy forwards ``stream: true`` upstream
    unchanged and relays each real chunk to the client as its own SSE frame,
    matching the upstream's own chunk boundaries.
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
                    assert resp.content_type == "text/event-stream"
                    raw = await resp.read()
        finally:
            await proxy.stop()

    assert received_stream_flags == [True], (
        f"proxy forwarded stream={received_stream_flags} upstream; "
        "must forward the real provider's own streaming shape unchanged"
    )
    frames = _parse_sse_body(raw)
    # 4 real upstream frames relayed live (role, 2x content, finish) plus one
    # spliced-in injection content frame before the (relayed) finish frame.
    content_deltas = [
        f["choices"][0]["delta"].get("content")
        for f in frames
        if f["choices"][0]["delta"].get("content")
    ]
    assert content_deltas == ["Real upstream ", "reply.", "\nPROXY-INJECTED-TEXT"], (
        "real content chunks must be relayed live, in order, with the "
        "injection appended as its own trailing delta - not merged/batched"
    )
    finish_frames = [f for f in frames if f["choices"][0].get("finish_reason")]
    assert len(finish_frames) == 1
    assert finish_frames[0]["choices"][0]["finish_reason"] == "stop"
    # The injection content must precede the finish_reason frame on the wire.
    assert frames.index(finish_frames[0]) == len(frames) - 1

    assert len(proxy.records) == 1
    assert proxy.records[0].response_text == "Real upstream reply.\nPROXY-INJECTED-TEXT"
    assert proxy.records[0].input_tokens == 3
    assert proxy.records[0].output_tokens == 2


@pytest.mark.asyncio
async def test_llm_proxy_relays_real_provider_sse_without_injection() -> None:
    """Same real-SSE-upstream shape as above but with no injection active:
    the proxy must be a transparent live relay (no batching, no extra
    frames), and recording must still assemble the full text correctly."""
    async with _real_streaming_upstream_stub() as (base_url, received_stream_flags):
        proxy = LLMProxy(
            upstream_base_url=base_url,
            upstream_api_key="sk-upstream",
            host="127.0.0.1",
            port=0,
        )
        await proxy.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{proxy.actual_port}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                ) as resp:
                    assert resp.content_type == "text/event-stream"
                    raw = await resp.read()
        finally:
            await proxy.stop()

    assert received_stream_flags == [True]
    frames = _parse_sse_body(raw)
    content_deltas = [
        f["choices"][0]["delta"].get("content")
        for f in frames
        if f["choices"][0]["delta"].get("content")
    ]
    assert content_deltas == ["Real upstream ", "reply."]
    assert proxy.records[0].response_text == "Real upstream reply."


@pytest.mark.asyncio
async def test_llm_proxy_falls_back_to_plain_json_when_client_does_not_want_stream() -> None:
    """If the inbound request explicitly opts out of streaming, the proxy
    must forward that intent upstream and return a single plain-JSON
    response, not SSE - the streaming relay path is only used when the
    caller (OpenClaw) actually asked for it."""
    async with _real_streaming_upstream_stub() as (base_url, received_stream_flags):
        proxy = LLMProxy(
            upstream_base_url=base_url,
            upstream_api_key="sk-upstream",
            host="127.0.0.1",
            port=0,
        )
        await proxy.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{proxy.actual_port}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                    },
                ) as resp:
                    assert resp.content_type == "application/json"
                    body = await resp.json()
        finally:
            await proxy.stop()
    assert received_stream_flags == [False]
    assert body["choices"][0]["message"]["content"] == "Real upstream reply."


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
    assert "model_system_prompt" in names
    assert MODEL_SYSTEM_PROMPT_CTRL.name == "model_system_prompt"


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
