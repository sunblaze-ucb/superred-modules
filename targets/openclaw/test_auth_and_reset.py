"""Unit tests for reviewer-fix areas: auth on host servers, reset RPC, proxy injection."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

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
