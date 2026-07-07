"""Opt-in Docker + real Gemini provider tests.

Mirrors the key scenarios from ``test_openclaw_provider_live.py`` but through
``managed_runtime=\"docker\"`` (production isolation path).

Run explicitly::

    GEMINI_API_KEY=... pytest test_docker_provider_live.py -v -m "provider and docker"
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from openclaw_target import OpenClawTarget
from openclaw_target.target import (
    FILE_CONTENT_CTRL,
    MODEL_RESPONSE_CTRL,
    MODEL_SYSTEM_PROMPT_CTRL,
    USER_MESSAGE_CTRL,
)
from test_support import docker_daemon_ready, docker_gemini_target, gemini_api_key

from superred.core.types.events import (
    ControllableInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
)


pytestmark = [
    pytest.mark.provider,
    pytest.mark.docker,
    pytest.mark.skipif(not docker_daemon_ready(), reason="Docker daemon unavailable"),
    pytest.mark.skipif(gemini_api_key() is None, reason="GEMINI_API_KEY not set"),
]


async def _run_with_user_message(
    target: OpenClawTarget,
    user_message: str,
) -> tuple[str | None, list[object]]:
    emitted: list[object] = []

    async def send_event(event: object) -> ControllableInjection:
        controllable = getattr(event, "controllable")
        value = user_message if controllable is USER_MESSAGE_CTRL else ""
        return ControllableInjection(
            event=event, controllable=controllable, value=value,  # type: ignore[arg-type]
        )

    await target.run(lambda e: emitted.append(e), send_event)
    return target.query("last_response"), emitted


@pytest.mark.asyncio
async def test_docker_provider_gemini_agent_turn() -> None:
    """Docker Target → real Gemini: basic agent turn completes."""
    target = docker_gemini_target(timeout_s=240)
    try:
        await target.warmup_static_observables()
        response, _ = await _run_with_user_message(
            target,
            "What is 17 + 25? Reply with only the numeric result.",
        )
        assert response is not None
        assert re.search(r"\b42\b", response), response
    finally:
        await target.teardown()


@pytest.mark.asyncio
async def test_docker_provider_gemini_response_injection_through_real_proxy() -> None:
    """Docker Target → proxy → real Gemini: response injection lands."""
    target = docker_gemini_target(timeout_s=240)
    try:
        await target.warmup_static_observables()

        async def send_event(event: object) -> ControllableInjection:
            controllable = getattr(event, "controllable")
            if controllable is MODEL_RESPONSE_CTRL:
                value = "DOCKER-PROVIDER-RESPONSE-INJECT-91"
            elif controllable is USER_MESSAGE_CTRL:
                value = "Say exactly: PROXY-OK"
            else:
                value = ""
            return ControllableInjection(
                event=event, controllable=controllable, value=value,  # type: ignore[arg-type]
            )

        await target.run(lambda _e: None, send_event)

        response = target.query("last_response")
        assert response is not None
        assert "DOCKER-PROVIDER-RESPONSE-INJECT-91" in response
        assert target._llm_proxy is not None
        assert target._llm_proxy.records
    finally:
        await target.teardown()


@pytest.mark.asyncio
async def test_docker_provider_gemini_streaming_through_real_proxy() -> None:
    """Docker Target → proxy → real Gemini: incremental assistant_stream."""
    target = docker_gemini_target(timeout_s=240)
    try:
        await target.warmup_static_observables()
        stream_chunks: list[str] = []

        def emit(event: object) -> None:
            if (
                isinstance(event, ObservableEvent)
                and event.observable.name == "assistant_stream"
                and event.content
            ):
                stream_chunks.append(event.content)

        async def send_event(event: object) -> ControllableInjection:
            controllable = getattr(event, "controllable")
            value = (
                "Write a 5-sentence story about a lighthouse keeper."
                if controllable is USER_MESSAGE_CTRL
                else ""
            )
            return ControllableInjection(
                event=event, controllable=controllable, value=value,  # type: ignore[arg-type]
            )

        await target.run(emit, send_event)

        assert target.query("last_response") is not None
        assert len(stream_chunks) >= 2, stream_chunks
    finally:
        await target.teardown()


@pytest.mark.asyncio
async def test_docker_provider_gemini_live_file_content_middleware_hook_fires() -> None:
    """Docker Target + real Gemini: live ``file_content`` middleware hook fires."""
    target = docker_gemini_target(timeout_s=240, enable_tool_injection=True)
    try:
        await target.warmup_static_observables()
        target.set_config(
            "workspace_files",
            json.dumps({"USER.md": "DOCKER-PROVIDER-ORIGINAL-93"}),
        )

        hook_calls: list[tuple[str, dict[str, Any]]] = []

        async def send_event(event: object) -> ControllableInjection:
            controllable = getattr(event, "controllable")
            if isinstance(event, ControllablePostCallEvent) and controllable is FILE_CONTENT_CTRL:
                hook_calls.append((controllable.name, json.loads(event.request)))
                return ControllableInjection(
                    event=event,  # type: ignore[arg-type]
                    controllable=controllable,
                    value="DOCKER-PROVIDER-INJECTED-FILE-93",
                )
            if isinstance(event, ControllablePreCallEvent) and controllable is USER_MESSAGE_CTRL:
                return ControllableInjection(
                    event=event,  # type: ignore[arg-type]
                    controllable=controllable,
                    value="Read USER.md and tell me what it says.",
                )
            return ControllableInjection(
                event=event, controllable=controllable, value="",  # type: ignore[arg-type]
            )

        await target.run(lambda _e: None, send_event)

        if not hook_calls:
            pytest.skip(
                "live model did not invoke read on this turn; hook path is "
                "validated in stub-LLM live tests",
            )
        assert hook_calls[0][0] == "file_content"
        assert hook_calls[0][1]["hook"] == "tool_result_middleware"
    finally:
        await target.teardown()
