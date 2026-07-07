"""Shared ``send_event`` builders and tool-injection assertions."""

from __future__ import annotations

import json
from typing import Any

from openclaw_target import OpenClawTarget
from openclaw_target.target import (
    FILE_CONTENT_CTRL,
    FILE_CONTENT_TRANSCRIPT_CTRL,
    USER_MESSAGE_CTRL,
)

from superred.core.types.events import (
    ControllableInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
)


def passthrough_send_event(
    user_message: str,
    hook_calls: list[tuple[str, dict[str, Any]]],
    *,
    file_injection: str = "INJECTED-BY-SUPERRED-42",
):
    """Answer ``user_message`` and inject transcript poison on ``read``."""

    async def send_event(event: object) -> ControllableInjection:
        controllable = getattr(event, "controllable")
        if (
            isinstance(event, ControllablePostCallEvent)
            and controllable is FILE_CONTENT_TRANSCRIPT_CTRL
        ):
            hook_calls.append((controllable.name, json.loads(event.request)))
            return ControllableInjection(
                event=event,  # type: ignore[arg-type]
                controllable=controllable,
                value=file_injection,
            )
        if isinstance(event, ControllablePreCallEvent) and controllable is USER_MESSAGE_CTRL:
            return ControllableInjection(
                event=event,  # type: ignore[arg-type]
                controllable=controllable,
                value=user_message,
            )
        return ControllableInjection(
            event=event, controllable=controllable, value="",  # type: ignore[arg-type]
        )

    return send_event


def live_file_send_event(
    user_message: str,
    hook_calls: list[tuple[str, dict[str, Any]]],
    *,
    file_injection: str = "INJECTED-BY-SUPERRED-42",
):
    """Answer ``user_message`` and inject live same-turn ``file_content``."""

    async def send_event(event: object) -> ControllableInjection:
        controllable = getattr(event, "controllable")
        if isinstance(event, ControllablePostCallEvent) and controllable is FILE_CONTENT_CTRL:
            hook_calls.append((controllable.name, json.loads(event.request)))
            return ControllableInjection(
                event=event,  # type: ignore[arg-type]
                controllable=controllable,
                value=file_injection,
            )
        if isinstance(event, ControllablePreCallEvent) and controllable is USER_MESSAGE_CTRL:
            return ControllableInjection(
                event=event,  # type: ignore[arg-type]
                controllable=controllable,
                value=user_message,
            )
        return ControllableInjection(
            event=event, controllable=controllable, value="",  # type: ignore[arg-type]
        )

    return send_event


def injecting_send_event(
    *,
    user_message: str,
    hook_calls: list[tuple[str, dict[str, Any]]],
    injections: dict[str, str],
) -> object:
    """Inject per-controllable values keyed by controllable ``name``."""

    async def send_event(event: object) -> ControllableInjection:
        controllable = getattr(event, "controllable")
        if isinstance(event, ControllablePostCallEvent):
            value = injections.get(controllable.name, "")
            if value:
                hook_calls.append((controllable.name, json.loads(event.request)))
            return ControllableInjection(
                event=event,  # type: ignore[arg-type]
                controllable=controllable,
                value=value,
            )
        if isinstance(event, ControllablePreCallEvent):
            value = injections.get(controllable.name, "")
            if controllable is USER_MESSAGE_CTRL and not value:
                value = user_message
            return ControllableInjection(
                event=event,  # type: ignore[arg-type]
                controllable=controllable,
                value=value,
            )
        return ControllableInjection(
            event=event, controllable=controllable, value="",  # type: ignore[arg-type]
        )

    return send_event


def assert_tool_injection_visible_same_turn(
    requests: list[dict[str, Any]],
    *,
    injection_marker: str,
    original_marker: str,
    request_index: int = 1,
) -> None:
    """Assert live middleware spoofing reached the in-flight continuation."""
    assert len(requests) > request_index
    messages = requests[request_index].get("messages", [])
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert tool_messages, (
        f"expected a tool message in continuation request: {messages}"
    )
    blob = json.dumps(tool_messages)
    assert injection_marker in blob, blob
    assert original_marker not in blob, blob


async def assert_tool_injection_persisted_on_next_run(
    target: OpenClawTarget,
    *,
    requests: list[dict[str, Any]],
    followup_user_message: str,
    hook_calls: list[tuple[str, dict[str, Any]]],
    injection_marker: str,
    original_marker: str,
    emitted: list[object],
) -> None:
    """Assert poisoned tool output appears on the next prompt submission."""
    requests_before = len(requests)
    hook_calls.clear()
    followup = injecting_send_event(
        user_message=followup_user_message,
        hook_calls=hook_calls,
        injections={},
    )
    await target.run(lambda e: emitted.append(e), followup)
    assert len(requests) > requests_before
    next_prompt_messages = requests[requests_before]["messages"]
    tool_messages = [m for m in next_prompt_messages if m.get("role") == "tool"]
    assert tool_messages, (
        f"expected a persisted tool message in history: {next_prompt_messages}"
    )
    assert any(injection_marker in str(m.get("content")) for m in tool_messages)
    assert all(
        original_marker not in str(m.get("content"))
        for m in tool_messages
    )
