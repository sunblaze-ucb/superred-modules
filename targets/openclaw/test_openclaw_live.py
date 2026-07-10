"""Live end-to-end tests against a real OpenClaw gateway process.

Unlike ``test_openclaw_integration.py`` (in-process ``MockGateway``), these
tests spawn the actual ``openclaw gateway`` CLI, connect over WebSocket, and
exercise real RPCs. A local stub HTTP server stands in for the upstream LLM
provider so no API keys are required.

Skipped unless the ``openclaw`` CLI is on ``PATH``. Docker is not required.

Run explicitly::

    pytest test_openclaw_live.py -v
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from openclaw_target import OpenClawTarget
from openclaw_target.device_identity import OPERATOR_SCOPES
from openclaw_target.proxy_llm import LLMProxy
from openclaw_target.runtime import OpenClawRuntime
from openclaw_target.ws_client import OpenClawWSClient
from test_support import (
    lan_ip,
    loopback_recording_stub_llm_server,
    loopback_stub_llm_server,
    loopback_stub_tool_calling_llm_server,
    openclaw_cli_ready,
    run_all_controllables_scenario,
    run_file_content_same_turn_scenario,
    run_managed_target_scenario,
    run_memory_poison_scenario,
    run_model_response_injection_scenario,
    run_model_system_prompt_injection_scenario,
    run_reset_teardown_scenario,
    run_shell_output_same_turn_scenario,
)

pytestmark = [
    pytest.mark.local_gateway,
    pytest.mark.skipif(not openclaw_cli_ready(), reason="openclaw CLI unavailable"),
]


@asynccontextmanager
async def _live_runtime(
    *,
    model_id: str = "openai/gpt-4o-mini",
    provider_base_url: str | None = None,
    provider_api_key: str = "",
) -> AsyncIterator[OpenClawRuntime]:
    rt = OpenClawRuntime(
        model_id=model_id,
        provider_base_url=provider_base_url,
        provider_api_key=provider_api_key,
    )
    await rt.start()
    try:
        yield rt
    finally:
        await rt.stop()


@pytest.mark.asyncio
async def test_live_gateway_lifecycle_and_rpc() -> None:
    """Start a real gateway, connect, and exercise core RPCs."""
    async with _live_runtime(model_id="openai/gpt-4o-mini") as rt:
        assert rt.auth_token
        client = OpenClawWSClient(
            gateway_url=rt.gateway_url,
            auth_token=rt.auth_token,
        )
        hello = await client.connect()
        assert hello.get("type") == "hello-ok"

        catalog = await client.rpc("tools.catalog")
        assert isinstance(catalog, dict)
        assert "profiles" in catalog

        await client.rpc(
            "agents.files.set",
            {
                "agentId": "main",
                "name": "AGENTS.md",
                "content": "# Live test\nCANARY-LIVE-001",
            },
        )

        await client.reset_session("live-rpc-session")

        await client.close()


@pytest.mark.asyncio
async def test_live_agent_run_with_stub_llm() -> None:
    """Drive a real agent turn through the gateway with a stub upstream."""
    async with loopback_stub_llm_server(reply="Live agent reply.") as stub_url:
        proxy = LLMProxy(
            upstream_base_url=stub_url,
            upstream_api_key="sk-stub",
            inbound_token="proxy-token",
        )
        await proxy.start()
        try:
            async with _live_runtime(
                provider_base_url=proxy.proxy_base_url,
                provider_api_key="proxy-token",
            ) as rt:
                client = OpenClawWSClient(
                    gateway_url=rt.gateway_url,
                    auth_token=rt.auth_token,
                )
                await client.connect()
                result = await client.run_agent(
                    "Say hello",
                    session_key="live-agent",
                    timeout_s=120,
                )
                assert result.status == "ok"
                assert result.error is None
                assert "Live agent reply." in result.assistant_text
                assert len(proxy.records) >= 1
                await client.close()
        finally:
            await proxy.stop()


@pytest.mark.asyncio
async def test_live_remote_path_grants_operator_scopes_via_device_identity() -> None:
    """Reproduce the Docker "remote" connect path without Docker.

    A ``lan``-bound gateway reached over a non-loopback IP is treated as a
    remote client, exactly like a containerised gateway reached over a
    published port. The device-less backend path yields ``scopes: []`` (and
    ``missing scope: operator.write`` on run); the device-identity path with a
    pre-seeded operator pairing must grant read/write/admin and allow the
    admin-scoped RPCs (``agents.files.set`` / ``sessions.reset``).
    """
    ip = lan_ip()
    if ip is None:
        pytest.skip("no non-loopback IPv4 available")

    rt = OpenClawRuntime(model_id="openai/gpt-4o-mini", bind="lan")
    await rt.start()
    try:
        assert rt.use_device_identity is True
        remote_url = f"ws://{ip}:{rt.host_port}"

        # Device-less backend path over a remote address: scopes cleared to [].
        backend = OpenClawWSClient(
            gateway_url=remote_url,
            auth_token=rt.auth_token or "",
            use_device_identity=False,
        )
        hello = await backend.connect()
        granted = hello.get("auth", {}).get("scopes") or hello.get("scopes") or []
        assert granted == [], f"expected empty scopes on backend remote path, got {granted}"
        await backend.close()

        # Device-identity path with pre-seeded pairing: full operator scopes.
        client = OpenClawWSClient(
            gateway_url=remote_url,
            auth_token=rt.auth_token or "",
            use_device_identity=rt.use_device_identity,
            device_identity_path=rt.device_identity_path,
        )
        hello = await client.connect()
        granted = hello.get("auth", {}).get("scopes") or hello.get("scopes") or []
        for scope in OPERATOR_SCOPES:
            assert scope in granted, f"missing {scope} in {granted}"

        # Admin-scoped RPCs must now succeed (file planting + session reset).
        await client.rpc(
            "agents.files.set",
            {"agentId": "main", "name": "AGENTS.md", "content": "# remote"},
        )
        await client.reset_session("remote-path")
        catalog = await client.rpc("tools.catalog")
        assert isinstance(catalog, dict)
        await client.close()
    finally:
        await rt.stop()


@pytest.mark.asyncio
async def test_live_file_content_same_turn_through_middleware() -> None:
    """``file_content`` rewrites the in-flight tool result via middleware."""
    async with loopback_stub_tool_calling_llm_server(
        tool_name="read",
        tool_arguments={"path": "USER.md"},
    ) as (stub_url, requests):
        target = OpenClawTarget(
            managed=True,
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="sk-stub",
            enable_tool_injection=True,
        )
        try:
            await run_file_content_same_turn_scenario(target, requests=requests)
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_live_memory_poison_round_trip() -> None:
    """``memory_poison`` enqueues next-turn context after the run."""
    async with loopback_recording_stub_llm_server(reply="Hello.") as (stub_url, requests):
        target = OpenClawTarget(
            managed=True,
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="sk-stub",
            enable_tool_injection=True,
        )
        try:
            await run_memory_poison_scenario(target, requests=requests)
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_live_model_response_injection_through_real_proxy() -> None:
    """``model_response_injection`` end to end through the real running proxy."""
    async with loopback_stub_llm_server(reply="Original stub reply.") as stub_url:
        target = OpenClawTarget(
            managed=True,
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="sk-stub",
        )
        try:
            await run_model_response_injection_scenario(target)
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_live_model_system_prompt_injection_through_real_proxy() -> None:
    """``model_system_prompt`` end to end: the real running proxy splices
    content into the system message the upstream provider receives."""
    async with loopback_recording_stub_llm_server(reply="Upstream ok.") as (stub_url, requests):
        target = OpenClawTarget(
            managed=True,
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="sk-stub",
        )
        try:
            await run_model_system_prompt_injection_scenario(
                target,
                injection_marker="LIVE-MODEL-SYSTEM-INJECT-55",
                requests=requests,
            )
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_live_shell_output_same_turn_through_middleware() -> None:
    """``shell_output`` live same-turn injection via middleware."""
    async with loopback_stub_tool_calling_llm_server(
        tool_name="exec",
        tool_arguments={"command": "echo ORIGINAL-EXEC-CONTENT-NOT-INJECTED"},
    ) as (stub_url, requests):
        target = OpenClawTarget(
            managed=True,
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="sk-stub",
            enable_tool_injection=True,
        )
        try:
            await run_shell_output_same_turn_scenario(target, requests=requests)
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_live_all_controllables_in_one_session() -> None:
    """One session exercising every controllable path against a real gateway.

    Pre-run: ``system_prompt_append`` config, ``model_system_prompt``,
    ``model_response_injection``, ``user_message``. Mid-run: ``file_content``
    on a real ``read`` tool call. End-of-run: ``memory_poison`` verified on
    the follow-up turn — all in one session.
    """
    async with loopback_stub_tool_calling_llm_server(
        tool_name="read",
        tool_arguments={"path": "USER.md"},
    ) as (stub_url, requests):
        target = OpenClawTarget(
            managed=True,
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="sk-stub",
            enable_tool_injection=True,
        )
        try:
            await run_all_controllables_scenario(
                target, marker_prefix="ALL-CTRL-LIVE", requests=requests,
            )
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_live_reset_and_teardown_against_real_gateway() -> None:
    """Reset/teardown semantics verified against a real gateway process."""
    async with loopback_stub_llm_server(reply="ok") as stub_url:
        target = OpenClawTarget(
            managed=True,
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="sk-stub",
            reset_session_between_runs=True,
        )
        try:
            await run_reset_teardown_scenario(target)
        finally:
            if target._runtime is not None:
                await target.teardown()


@pytest.mark.asyncio
async def test_live_openclaw_target_managed_run() -> None:
    """Full OpenClawTarget managed pipeline against a real gateway."""
    async with loopback_stub_llm_server(reply="Managed target live reply.") as stub_url:
        target = OpenClawTarget(
            managed=True,
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="sk-stub",
        )
        try:
            await run_managed_target_scenario(target)
        finally:
            await target.teardown()
