from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from superred.core.channel import EventEnvelope
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import ObservableEvent
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag

USER_TAG = SecurityDomainTag("user")
SYSTEM_TAG = SecurityDomainTag("system")
PROMPT_TAG = SecurityDomainTag("prompt", parent=SYSTEM_TAG)
MODEL_TAG = SecurityDomainTag("model_identity", parent=SYSTEM_TAG)
TOOL_CATALOG_TAG = SecurityDomainTag("tool_catalogue", parent=SYSTEM_TAG)
TOOL_CATALOG_READABLE_TAG = SecurityDomainTag(
    "tool_catalogue_readable", parent=TOOL_CATALOG_TAG
)
TOOL_CATALOG_ADDABLE_TAG = SecurityDomainTag(
    "tool_catalogue_addable", parent=TOOL_CATALOG_TAG
)
TOOLS_TAG = SecurityDomainTag("tools")
AGENT_TRACE_TAG = SecurityDomainTag("agent_trace", parent=SYSTEM_TAG)


class FakeReadableTrajectory:
    def __init__(self) -> None:
        self._items: list[Any] = []

    def push(self, item: Any) -> None:
        self._items.append(item)

    def snapshot(self) -> list[Any]:
        return list(self._items)

    def drain(self) -> list[Any]:
        items = list(self._items)
        self._items = []
        return items


def make_controllable(
    name: str = "user_prompt",
    tag: SecurityDomainTag = USER_TAG,
    *,
    value_type: str = "text",
) -> Controllable:
    return Controllable(name=name, security_domain=tag, value_type=value_type)


def make_observable(
    name: str,
    tag: SecurityDomainTag,
    *,
    observable_type: str = "text",
) -> Observable:
    return Observable(name=name, security_domain=tag, observable_type=observable_type)


def make_observable_value(
    name: str,
    content: Any,
    tag: SecurityDomainTag,
    *,
    observable_type: str = "text",
) -> ObservableValue:
    return ObservableValue(
        make_observable(name, tag, observable_type=observable_type), content
    )


def observable_event(
    name: str, content: Any, tag: SecurityDomainTag = AGENT_TRACE_TAG
) -> ObservableEvent:
    return ObservableEvent(observable=make_observable(name, tag), content=content)


class MockChoice:
    def __init__(self, content: str) -> None:
        self.message = type("Message", (), {"content": content})()


class MockResponse:
    def __init__(self, content: str) -> None:
        self.choices = [MockChoice(content)]


def mock_response(content: str) -> MockResponse:
    return MockResponse(content)


def empty_llm() -> AsyncMock:
    llm = AsyncMock()

    async def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("LLM should not be called in this test")

    llm.complete.side_effect = fail
    return llm


async def dispatch_event(opt: Any, event: Event) -> EventResponse:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[EventResponse] = loop.create_future()
    envelope = EventEnvelope(event=event, future=future, loop=loop)
    try:
        await opt._dispatch(envelope)
    except Exception:
        await asyncio.sleep(0)
        if future.done():
            future.exception()
        raise
    return await future
