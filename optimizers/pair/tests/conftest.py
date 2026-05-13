"""Shared test helpers for PAIR tests."""

from __future__ import annotations

from typing import Any
import asyncio

from superred.core.channel import EventEnvelope
from superred.core.interfaces.optimizer import Optimizer
from superred.core.types.event import Event, EventResponse
from unittest.mock import MagicMock

from superred.core.types.controllable import Controllable
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag

USER_TAG = SecurityDomainTag("user")
MODEL_TAG = SecurityDomainTag("model")
RESPONSE_TAG = SecurityDomainTag("response_readable")
SYSTEM_PROMPT_TAG = SecurityDomainTag("system_prompt")
SYSTEM_PROMPT_READABLE_TAG = SecurityDomainTag("system_prompt_readable")
MODEL_IDENTITY_TAG = SecurityDomainTag("model_identity")


def mock_response(content: str | None) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def make_controllable(
    name: str = "user_message", tag: SecurityDomainTag = USER_TAG
) -> Controllable:
    return Controllable(name=name, security_domain=tag, value_type="text")


def make_observable(name: str = "response", tag: SecurityDomainTag = RESPONSE_TAG) -> Observable:
    return Observable(name=name, security_domain=tag)


def observable_value(
    name: str, content: str, tag: SecurityDomainTag = MODEL_IDENTITY_TAG
) -> ObservableValue:
    return ObservableValue(observable=make_observable(name, tag), content=content)


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


async def dispatch_event(optimizer: Optimizer, event: Event) -> EventResponse:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[EventResponse] = loop.create_future()
    envelope = EventEnvelope(event=event, future=future, loop=loop)
    try:
        await optimizer._dispatch(envelope)
    except Exception:
        await asyncio.sleep(0)
        if future.done():
            future.exception()
        raise
    return await future
