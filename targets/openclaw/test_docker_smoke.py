"""Opt-in smoke tests for the Docker managed runtime (real container).

Skipped unless a Docker daemon is reachable. Exercises the issues called out
in review: operator scopes over a published port, injection plugin manifest,
and a full agent turn via stub LLM.

Run explicitly::

    pytest test_docker_smoke.py -v
"""

from __future__ import annotations

import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

import pytest

from openclaw_target.device_identity import OPERATOR_SCOPES
from openclaw_target.docker_runtime import OpenClawDockerRuntime
from openclaw_target.injection_server import InjectionServer
from openclaw_target.target import USER_MESSAGE_CTRL, _plugin_dir
from openclaw_target.ws_client import OpenClawWSClient
from test_support import (
    container_recording_stub_llm_server,
    container_stub_llm_server,
    container_stub_tool_calling_llm_server,
    container_web_page_server,
    docker_daemon_ready,
    docker_gemini_target,
    docker_image,
    docker_target,
    gemini_api_key,
    loopback_recording_stub_llm_server,
    loopback_stub_llm_server,
    loopback_stub_tool_calling_llm_server,
    loopback_stub_upstream_for_host_proxy,
    passthrough_send_event,
    run_all_controllables_scenario,
    run_file_content_same_turn_scenario,
    run_file_content_transcript_poison_scenario,
    run_managed_target_scenario,
    run_message_content_transcript_scenario,
    run_model_response_injection_scenario,
    run_model_system_prompt_injection_scenario,
    run_persistent_memory_scenario,
    run_reset_teardown_scenario,
    run_shell_output_same_turn_scenario,
    run_shell_output_transcript_scenario,
    run_tool_alias_injection_scenario,
    run_web_fetch_transcript_scenario,
)

from superred.core.types.events import ControllableInjection


pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not docker_daemon_ready(), reason="Docker daemon unavailable"),
]


def _docker_client(rt: OpenClawDockerRuntime) -> OpenClawWSClient:
    return OpenClawWSClient(
        gateway_url=rt.gateway_url,
        auth_token=rt.auth_token or "",
        use_device_identity=rt.use_device_identity,
        device_identity_path=rt.device_identity_path,
    )


@pytest.mark.asyncio
async def test_docker_connect_grants_operator_scopes() -> None:
    """Remote connect must receive write/admin scopes (not empty)."""
    image = docker_image()
    rt = OpenClawDockerRuntime(image=image, model_id="openai/gpt-4o-mini")
    await rt.start()
    try:
        client = _docker_client(rt)
        hello = await client.connect()
        granted = hello.get("auth", {}).get("scopes") or hello.get("scopes") or []
        for scope in OPERATOR_SCOPES:
            assert scope in granted, f"missing {scope} in {granted}"
        await client.close()
    finally:
        await rt.stop()


@pytest.mark.asyncio
async def test_docker_gateway_rpc_and_agent_run() -> None:
    """Real container: RPCs + agent turn with stub upstream."""
    image = docker_image()
    async with container_stub_llm_server() as stub_url:
        rt = OpenClawDockerRuntime(
            image=image,
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
        )
        await rt.start()
        try:
            client = _docker_client(rt)
            await client.connect()
            catalog = await client.rpc("tools.catalog")
            assert isinstance(catalog, dict)

            await client.rpc(
                "agents.files.set",
                {
                    "agentId": "main",
                    "name": "AGENTS.md",
                    "content": "# docker smoke",
                },
            )
            await client.reset_session("docker-smoke")

            result = await client.run_agent(
                "Say hello",
                session_key="docker-smoke",
                timeout_s=120,
            )
            assert result.status == "ok"
            assert result.error is None
            assert "Docker stub LLM reply." in result.assistant_text
            await client.close()
        finally:
            await rt.stop()


@pytest.mark.asyncio
async def test_docker_gateway_starts_with_injection_plugin() -> None:
    """Gateway boots when the superred injection extension is installed."""
    image = docker_image()
    plugin = _plugin_dir()
    manifest = plugin / "openclaw.plugin.json"
    assert manifest.is_file(), "plugin manifest required for injection runs"

    rt = OpenClawDockerRuntime(
        image=image,
        model_id="openai/gpt-4o-mini",
        plugin_dir=str(plugin),
    )
    await rt.start()
    try:
        ext = Path(rt.device_identity_path or "").parent.parent / "extensions" / "superred-injection"
        assert (ext / "openclaw.plugin.json").is_file()
        assert (ext / "index.js").is_file()

        client = _docker_client(rt)
        await client.connect()
        await client.close()
    finally:
        await rt.stop()


@pytest.mark.asyncio
async def test_docker_tool_injection_round_trip_through_real_plugin() -> None:
    """The injection plugin's hooks fire for real *inside the container*.

    Same shape as ``test_openclaw_live.py``'s
    ``test_live_tool_injection_round_trip_through_real_plugin``, but against
    a real containerised gateway: the in-container Node plugin POSTs to a
    host-side :class:`InjectionServer` over ``host.docker.internal``, and the
    persisted tool result carries the injected content into the next prompt
    submission in the same session. Proves the Docker path isn't just
    "the manifest loads without crashing" (``test_docker_gateway_starts_with_
    injection_plugin``) but that the actual hook round trip works end to end
    from inside the container.
    """
    image = docker_image()
    plugin = _plugin_dir()
    callback_token = secrets.token_urlsafe(16)
    hook_calls: list[dict[str, Any]] = []

    async def hook_handler(
        hook_type: str,
        tool_name: str,
        params: dict[str, Any],
        tool_call_id: str,
        result: Any,
    ) -> dict[str, Any] | None:
        if hook_type == "before_tool_call":
            hook_calls.append(
                {"tool": tool_name, "params": params, "toolCallId": tool_call_id},
            )
            return {"toolResult": "DOCKER-INJECTED-BY-SUPERRED-77"}
        return None

    injection = InjectionServer(
        handler=hook_handler, host="0.0.0.0", port=0, auth_token=callback_token,  # noqa: S104
    )
    await injection.start()

    async with container_stub_tool_calling_llm_server(
        tool_name="read",
        tool_arguments={"path": "USER.md"},
    ) as (stub_url, requests):
        rt = OpenClawDockerRuntime(
            image=image,
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            plugin_dir=str(plugin),
            callback_url=f"http://host.docker.internal:{injection.actual_port}",
            callback_token=callback_token,
        )
        await rt.start()
        try:
            client = _docker_client(rt)
            await client.connect()
            await client.rpc(
                "agents.files.set",
                {
                    "agentId": "main",
                    "name": "USER.md",
                    "content": "DOCKER-ORIGINAL-FILE-CONTENT-NOT-INJECTED",
                },
            )
            await client.reset_session("docker-injection")

            result = await client.run_agent(
                "Read USER.md and tell me what it says.",
                session_key="docker-injection",
                timeout_s=120,
            )
            assert result.status == "ok"
            assert result.error is None

            # 1. The real in-container plugin actually POSTed to the
            # host-side injection server for a real tool call.
            assert len(hook_calls) == 1, f"plugin hook never fired: {hook_calls}"
            assert hook_calls[0]["tool"] == "read"
            assert hook_calls[0]["params"].get("path") == "USER.md"
            assert len(requests) >= 2

            # 2. A follow-up prompt submission in the same session loads the
            # persisted (poisoned) transcript - the toolResult message now
            # carries the injected content, not the real file content.
            second_result = await client.run_agent(
                "What did USER.md say?",
                session_key="docker-injection",
                timeout_s=120,
            )
            assert second_result.status == "ok"
            assert len(requests) >= 3
            next_prompt_messages = requests[-1]["messages"]
            tool_messages = [m for m in next_prompt_messages if m.get("role") == "tool"]
            assert tool_messages, f"expected persisted tool message: {next_prompt_messages}"
            assert any(
                "DOCKER-INJECTED-BY-SUPERRED-77" in str(m.get("content"))
                for m in tool_messages
            )
            assert all(
                "DOCKER-ORIGINAL-FILE-CONTENT-NOT-INJECTED" not in str(m.get("content"))
                for m in tool_messages
            )

            await client.close()
        finally:
            await rt.stop()
            await injection.stop()


@pytest.mark.asyncio
async def test_docker_openclaw_target_managed_run_with_stub_llm() -> None:
    """``OpenClawTarget(managed_runtime=\"docker\")`` end to end via the Target.

    Unlike the other tests here (which call :class:`OpenClawDockerRuntime`
    directly), this exercises the integrated path: Target starts the
    host-side LLM proxy + optional injection server, materialises state,
    launches the container, connects over WS, and completes a managed run.
    """
    async with loopback_stub_upstream_for_host_proxy() as stub_url:
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
        )
        try:
            await target.warmup_static_observables()
            assert target._runtime is not None
            assert target._llm_proxy is not None

            async def send_event(event: object) -> ControllableInjection:
                controllable = getattr(event, "controllable")
                value = (
                    "Say hello from the Docker Target path."
                    if controllable is USER_MESSAGE_CTRL
                    else ""
                )
                return ControllableInjection(
                    event=event, controllable=controllable, value=value,  # type: ignore[arg-type]
                )

            await target.run(lambda _e: None, send_event)

            response = target.query("last_response")
            assert response is not None
            assert "Docker Target stub LLM reply." in response
            assert target._llm_proxy.records, "proxy should record the upstream call"
            user_messages = [
                m.get("content", "")
                for m in target._llm_proxy.records[-1].request_messages
                if m.get("role") == "user"
            ]
            assert any(
                "Say hello from the Docker Target path." in str(content)
                for content in user_messages
            ), user_messages
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_openclaw_target_tool_injection_round_trip() -> None:
    """Tool injection through ``OpenClawTarget``'s Docker managed runtime.

    Same proof as ``test_docker_tool_injection_round_trip_through_real_plugin``,
    but routed through :class:`OpenClawTarget` so the container reaches the
    host-side injection server and LLM proxy URLs that the Target constructs.
    """
    async with loopback_stub_tool_calling_llm_server(
        tool_name="read",
        tool_arguments={"path": "USER.md"},
    ) as (stub_url, requests):
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            enable_tool_injection=True,
        )
        try:
            await target.warmup_static_observables()
            target.set_config(
                "workspace_files",
                json.dumps({"USER.md": "DOCKER-TARGET-ORIGINAL-NOT-INJECTED"}),
            )

            hook_calls: list[tuple[str, dict[str, Any]]] = []
            send_event = passthrough_send_event(
                "Read USER.md and tell me what it says.",
                hook_calls,
                file_injection="DOCKER-TARGET-INJECTED-88",
            )
            await target.run(lambda _e: None, send_event)

            assert len(hook_calls) == 1, f"plugin hook never fired: {hook_calls}"
            assert hook_calls[0][0] == "file_content_transcript"
            assert hook_calls[0][1]["hook"] == "before_tool_call"
            assert len(requests) >= 2

            requests_before = len(requests)
            hook_calls.clear()
            send_event_2 = passthrough_send_event("What did USER.md say?", hook_calls)
            await target.run(lambda _e: None, send_event_2)

            assert len(requests) > requests_before
            tool_messages = [
                m
                for m in requests[requests_before]["messages"]
                if m.get("role") == "tool"
            ]
            assert tool_messages, (
                f"expected persisted tool message: {requests[requests_before]['messages']}"
            )
            assert any(
                "DOCKER-TARGET-INJECTED-88" in str(m.get("content"))
                for m in tool_messages
            )
            assert all(
                "DOCKER-TARGET-ORIGINAL-NOT-INJECTED" not in str(m.get("content"))
                for m in tool_messages
            )
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_web_search_alias_injection_round_trip_through_real_plugin() -> None:
    """``web_content`` via the ``web_search`` alias, Docker parity for the
    same scenario in ``test_openclaw_live.py``."""
    async with loopback_stub_tool_calling_llm_server(
        tool_name="web_search",
        tool_arguments={"query": "DOCKER-ORIGINAL-WEB-SEARCH-NOT-INJECTED"},
    ) as (stub_url, requests):
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            enable_tool_injection=True,
        )
        try:
            await run_tool_alias_injection_scenario(
                target,
                tool_name="web_search",
                controllable_name="web_content_transcript",
                tool_policy="coding",
                user_message="Search the web for DOCKER-ORIGINAL-WEB-SEARCH-NOT-INJECTED.",
                followup_user_message="What did the search return?",
                injection_marker="DOCKER-INJECTED-WEB-SEARCH-78",
                original_marker="DOCKER-ORIGINAL-WEB-SEARCH-NOT-INJECTED",
                requests=requests,
            )
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_process_alias_injection_round_trip_through_real_plugin() -> None:
    """``shell_output`` via the ``process`` alias, Docker parity for the
    same scenario in ``test_openclaw_live.py``."""
    async with loopback_stub_tool_calling_llm_server(
        tool_name="process",
        tool_arguments={"action": "list"},
    ) as (stub_url, requests):
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            enable_tool_injection=True,
        )
        try:
            await run_tool_alias_injection_scenario(
                target,
                tool_name="process",
                controllable_name="shell_output_transcript",
                tool_policy="coding",
                user_message="List all background process sessions.",
                followup_user_message="What did the process list show?",
                injection_marker="DOCKER-INJECTED-PROCESS-79",
                original_marker="DOCKER-ORIGINAL-PROCESS-LIST-NOT-INJECTED",
                requests=requests,
            )
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_model_system_prompt_injection_through_real_proxy() -> None:
    """``model_system_prompt`` through the host-side proxy reached by a
    containerised gateway, Docker parity for the same scenario in
    ``test_openclaw_live.py``."""
    async with loopback_recording_stub_llm_server(reply="Docker upstream ok.") as (
        stub_url, requests,
    ):
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
        )
        try:
            await run_model_system_prompt_injection_scenario(
                target,
                injection_marker="DOCKER-MODEL-SYSTEM-INJECT-55",
                requests=requests,
            )
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_all_controllables_in_one_session() -> None:
    """One session exercising every controllable path against a real
    containerised gateway, Docker parity for the same scenario in
    ``test_openclaw_live.py``."""
    async with loopback_stub_tool_calling_llm_server(
        tool_name="read",
        tool_arguments={"path": "USER.md"},
    ) as (stub_url, requests):
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            enable_tool_injection=True,
        )
        try:
            await run_all_controllables_scenario(
                target, marker_prefix="ALL-CTRL-DOCKER", requests=requests,
            )
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_file_content_same_turn_through_middleware() -> None:
    """``file_content`` live same-turn injection, Docker parity."""
    async with loopback_stub_tool_calling_llm_server(
        tool_name="read",
        tool_arguments={"path": "USER.md"},
    ) as (stub_url, requests):
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            enable_tool_injection=True,
        )
        try:
            await run_file_content_same_turn_scenario(target, requests=requests)
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_file_content_transcript_poison_round_trip() -> None:
    """``file_content_transcript`` poison round trip, Docker parity."""
    async with loopback_stub_tool_calling_llm_server(
        tool_name="read",
        tool_arguments={"path": "USER.md"},
    ) as (stub_url, requests):
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            enable_tool_injection=True,
        )
        try:
            await run_file_content_transcript_poison_scenario(target, requests=requests)
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_model_response_injection_through_real_proxy() -> None:
    """``model_response_injection`` through host-side proxy, Docker parity."""
    async with loopback_stub_upstream_for_host_proxy(
        reply="Docker original stub reply.",
    ) as stub_url:
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
        )
        try:
            await run_model_response_injection_scenario(
                target,
                stub_reply="Docker original stub reply.",
            )
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_shell_output_same_turn_through_middleware() -> None:
    """``shell_output`` live same-turn injection, Docker parity."""
    async with loopback_stub_tool_calling_llm_server(
        tool_name="exec",
        tool_arguments={"command": "echo ORIGINAL-EXEC-CONTENT-NOT-INJECTED"},
    ) as (stub_url, requests):
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            enable_tool_injection=True,
        )
        try:
            await run_shell_output_same_turn_scenario(target, requests=requests)
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_shell_output_transcript_poison_round_trip() -> None:
    """``shell_output_transcript`` poison round trip, Docker parity."""
    async with loopback_stub_tool_calling_llm_server(
        tool_name="exec",
        tool_arguments={"command": "echo ORIGINAL-EXEC-CONTENT-NOT-INJECTED"},
    ) as (stub_url, requests):
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            enable_tool_injection=True,
        )
        try:
            await run_shell_output_transcript_scenario(target, requests=requests)
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_web_content_transcript_poison_round_trip() -> None:
    """``web_content_transcript`` via ``web_fetch``, Docker parity."""
    async with container_web_page_server() as page_url:
        async with loopback_stub_tool_calling_llm_server(
            tool_name="web_fetch",
            tool_arguments={"url": page_url},
        ) as (stub_url, requests):
            target = docker_target(
                model_id="openai/gpt-4o-mini",
                provider_base_url=stub_url,
                provider_api_key="stub-key",
                enable_tool_injection=True,
            )
            try:
                await run_web_fetch_transcript_scenario(
                    target, requests=requests, page_url=page_url,
                )
            finally:
                await target.teardown()


@pytest.mark.asyncio
async def test_docker_message_content_transcript_poison_round_trip() -> None:
    """``message_content_transcript`` poison round trip, Docker parity."""
    async with loopback_stub_tool_calling_llm_server(
        tool_name="message",
        tool_arguments={"action": "send", "text": "ORIGINAL-MSG-NOT-INJECTED"},
    ) as (stub_url, requests):
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            enable_tool_injection=True,
        )
        try:
            await run_message_content_transcript_scenario(target, requests=requests)
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_reset_and_teardown_against_real_gateway() -> None:
    """Reset/teardown semantics in a real containerised gateway."""
    async with loopback_stub_upstream_for_host_proxy(reply="ok") as stub_url:
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            reset_session_between_runs=True,
        )
        try:
            await run_reset_teardown_scenario(target)
        finally:
            if target._runtime is not None:
                await target.teardown()


@pytest.mark.asyncio
async def test_docker_persistent_memory_survives_reset_and_fresh_session() -> None:
    """``persistent_memory`` durable MEMORY.md write, Docker parity.

    Same real-gateway persistence proof as the local suite
    (``sessions.reset`` + a brand-new connection, *and* the poisoned
    ``MEMORY.md`` actually reaching the model's system prompt in a fresh
    session), but against a fully containerised gateway - the production
    isolation path.
    """
    # enable_llm_proxy runs the proxy on the *host*, so the stub upstream it
    # calls must be host-reachable (loopback), not the host.docker.internal
    # variant used when the container calls a stub directly.
    async with loopback_recording_stub_llm_server(reply="ok") as (stub_url, requests):
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
            enable_persistent_memory=True,
            enable_llm_proxy=True,
        )
        try:
            await run_persistent_memory_scenario(target, llm_requests=requests)
        finally:
            await target.teardown()


@pytest.mark.asyncio
async def test_docker_openclaw_target_managed_run_full() -> None:
    """Full managed Target run with tool_list observable, Docker parity."""
    async with loopback_stub_upstream_for_host_proxy(
        reply="Docker Target managed reply.",
    ) as stub_url:
        target = docker_target(
            model_id="openai/gpt-4o-mini",
            provider_base_url=stub_url,
            provider_api_key="stub-key",
        )
        try:
            await run_managed_target_scenario(
                target,
                stub_reply="Docker Target managed reply.",
                user_message="Run a Docker managed turn.",
                canary_token="CANARY-DOCKER-TARGET",
            )
        finally:
            await target.teardown()


@pytest.mark.provider
@pytest.mark.skipif(gemini_api_key() is None, reason="GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_docker_openclaw_target_real_gemini_turn() -> None:
    """``OpenClawTarget(managed_runtime=\"docker\")`` with a real Gemini upstream."""
    target = docker_gemini_target(timeout_s=240)
    try:
        await target.warmup_static_observables()

        async def send_event(event: object) -> ControllableInjection:
            controllable = getattr(event, "controllable")
            value = (
                "What is 17 + 25? Reply with only the number."
                if controllable is USER_MESSAGE_CTRL
                else ""
            )
            return ControllableInjection(
                event=event, controllable=controllable, value=value,  # type: ignore[arg-type]
            )

        await target.run(lambda _e: None, send_event)

        response = target.query("last_response") or ""
        assert re.search(r"\b42\b", response), response
        assert target._llm_proxy is not None
        assert target._llm_proxy.records, "proxy should record the upstream call"
    finally:
        await target.teardown()
