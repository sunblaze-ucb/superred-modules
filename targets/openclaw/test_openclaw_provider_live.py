"""Opt-in live tests against a real LLM provider (Gemini).

Unlike ``test_openclaw_live.py`` (stub upstream), these exercises use a real
provider API. They are skipped unless both the ``openclaw`` CLI and
``GEMINI_API_KEY`` are set. Never commit API keys — pass them via environment
only.

Run explicitly::

    GEMINI_API_KEY=... pytest test_openclaw_provider_live.py -v

Optional overrides::

    OPENCLAW_PROVIDER_MODEL=google/gemini-2.5-flash
    OPENCLAW_PROVIDER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/v1
"""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any

import pytest

from openclaw_target import OpenClawTarget
from openclaw_target.target import (
    FILE_CONTENT_CTRL,
    MODEL_RESPONSE_CTRL,
    MODEL_SYSTEM_PROMPT_CTRL,
    USER_MESSAGE_CTRL,
)

from superred.core.types.events import (
    ControllableInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
)

DEFAULT_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/v1"
DEFAULT_GEMINI_MODEL = "google/gemini-2.5-flash"
_PROVIDER_TIMEOUT_S = 180


def _openclaw_cli_ready() -> bool:
    return shutil.which("openclaw") is not None


def _gemini_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key or None


def _provider_model() -> str:
    return os.environ.get("OPENCLAW_PROVIDER_MODEL", DEFAULT_GEMINI_MODEL)


def _provider_base_url() -> str:
    return os.environ.get("OPENCLAW_PROVIDER_BASE_URL", DEFAULT_GEMINI_BASE)


pytestmark = [
    pytest.mark.provider,
    pytest.mark.skipif(not _openclaw_cli_ready(), reason="openclaw CLI unavailable"),
    pytest.mark.skipif(_gemini_api_key() is None, reason="GEMINI_API_KEY not set"),
]


def _gemini_target(**kwargs: Any) -> OpenClawTarget:
    key = _gemini_api_key()
    assert key is not None
    return OpenClawTarget(
        managed=True,
        model_id=_provider_model(),
        provider_base_url=_provider_base_url(),
        provider_api_key=key,
        agent_timeout_s=_PROVIDER_TIMEOUT_S,
        **kwargs,
    )


async def _run_with_user_message(
    target: OpenClawTarget,
    user_message: str,
    *,
    send_event: object | None = None,
) -> tuple[str | None, list[object]]:
    emitted: list[object] = []

    async def default_send_event(event: object) -> ControllableInjection:
        controllable = getattr(event, "controllable")
        value = user_message if controllable is USER_MESSAGE_CTRL else ""
        return ControllableInjection(
            event=event, controllable=controllable, value=value,  # type: ignore[arg-type]
        )

    await target.run(
        lambda e: emitted.append(e),
        send_event or default_send_event,
    )
    return target.query("last_response"), emitted


@pytest.mark.asyncio
async def test_provider_direct_gemini_agent_turn() -> None:
    """Real gateway → real Gemini (no proxy): basic agent turn completes."""
    target = _gemini_target(enable_llm_proxy=False)
    try:
        await target.warmup_static_observables()
        response, _ = await _run_with_user_message(
            target,
            "What is 17 + 25? Reply with only the numeric result, no explanation.",
        )
        assert response is not None
        assert re.search(r"\b42\b", response), response
    finally:
        await target.teardown()


@pytest.mark.asyncio
async def test_provider_gemini_response_injection_through_real_proxy() -> None:
    """Real gateway → real proxy → real Gemini: response injection lands."""
    target = _gemini_target()
    try:
        await target.warmup_static_observables()

        async def send_event(event: object) -> ControllableInjection:
            controllable = getattr(event, "controllable")
            if controllable is MODEL_RESPONSE_CTRL:
                value = "PROVIDER-LIVE-RESPONSE-INJECT-91"
            elif controllable is USER_MESSAGE_CTRL:
                value = "Say exactly: PROXY-OK"
            else:
                value = ""
            return ControllableInjection(
                event=event, controllable=controllable, value=value,  # type: ignore[arg-type]
            )

        emitted: list[object] = []
        await target.run(lambda e: emitted.append(e), send_event)

        response = target.query("last_response")
        assert response is not None
        assert "PROVIDER-LIVE-RESPONSE-INJECT-91" in response

        model_resp_events = [
            e for e in emitted
            if isinstance(e, ObservableEvent)
            and e.observable.name == "model_response"
        ]
        assert model_resp_events
        assert "PROVIDER-LIVE-RESPONSE-INJECT-91" in model_resp_events[-1].content

        assert target._llm_proxy is not None
        assert target._llm_proxy.records, "proxy should record the upstream call"
    finally:
        await target.teardown()


@pytest.mark.asyncio
async def test_provider_gemini_system_prompt_injection_through_real_proxy() -> None:
    """Real proxy splices ``model_system_prompt`` into upstream requests."""
    target = _gemini_target()
    try:
        await target.warmup_static_observables()

        async def send_event(event: object) -> ControllableInjection:
            controllable = getattr(event, "controllable")
            if controllable is MODEL_SYSTEM_PROMPT_CTRL:
                value = "PROVIDER-LIVE-SYSTEM-INJECT-92"
            elif controllable is USER_MESSAGE_CTRL:
                value = "Reply with exactly: SYSTEM-OK"
            else:
                value = ""
            return ControllableInjection(
                event=event, controllable=controllable, value=value,  # type: ignore[arg-type]
            )

        await target.run(lambda e: None, send_event)

        assert target._llm_proxy is not None
        assert target._llm_proxy.records, "proxy should record the upstream call"
        recorded = target._llm_proxy.records[-1]
        system_messages = [
            m.get("content", "")
            for m in recorded.request_messages
            if m.get("role") == "system"
        ]
        assert system_messages
        assert any(
            "PROVIDER-LIVE-SYSTEM-INJECT-92" in str(content)
            for content in system_messages
        ), system_messages
    finally:
        await target.teardown()


@pytest.mark.asyncio
async def test_provider_gemini_streaming_through_real_proxy() -> None:
    """Real Gemini SSE via proxy: ``assistant_stream`` is incremental, not batched."""
    target = _gemini_target()
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
        assert len(stream_chunks) >= 2, (
            "expected multiple live assistant_stream deltas through the real "
            f"provider/proxy path, got {len(stream_chunks)} chunk(s): "
            f"{stream_chunks!r}"
        )
    finally:
        await target.teardown()


@pytest.mark.asyncio
async def test_provider_gemini_tool_injection_plugin_hook_fires() -> None:
    """Real Gemini chooses a tool; real plugin hook fires on ``before_tool_call``."""
    target = _gemini_target(enable_tool_injection=True)
    try:
        await target.warmup_static_observables()
        target.set_config(
            "workspace_files",
            json.dumps({"USER.md": "PROVIDER-ORIGINAL-FILE-CONTENT-93"}),
        )

        hook_calls: list[tuple[str, dict[str, Any]]] = []

        async def send_event(event: object) -> ControllableInjection:
            controllable = getattr(event, "controllable")
            if isinstance(event, ControllablePostCallEvent) and controllable is FILE_CONTENT_CTRL:
                hook_calls.append((controllable.name, json.loads(event.request)))
                return ControllableInjection(
                    event=event,  # type: ignore[arg-type]
                    controllable=controllable,
                    value="PROVIDER-INJECTED-FILE-93",
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

        await target.run(lambda e: None, send_event)

        if not hook_calls:
            pytest.skip(
                "live model did not invoke read on this turn; hook path is "
                "validated in stub-LLM live tests",
            )
        assert hook_calls[0][0] == "file_content"
        assert hook_calls[0][1]["tool"] == "read"
        assert hook_calls[0][1]["params"].get("path") == "USER.md"
    finally:
        await target.teardown()
