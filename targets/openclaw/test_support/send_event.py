"""Shared ``send_event`` builders and tool-injection assertions."""

from __future__ import annotations

import json
from typing import Any

from openclaw_target import OpenClawTarget
from openclaw_target.target import FILE_CONTENT_CTRL, USER_MESSAGE_CTRL

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
    """Answer ``user_message`` and inject a fixed ``file_content`` value."""

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
            hook_calls.append((controllable.name, json.loads(event.request)))
            value = injections.get(controllable.name, "")
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
