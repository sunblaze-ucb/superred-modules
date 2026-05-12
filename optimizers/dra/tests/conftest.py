"""Shared DRA test helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from superred.core.types.controllable import Controllable
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag

USER_TAG = SecurityDomainTag("user")
MODEL_TAG = SecurityDomainTag("model")
RESPONSE_TAG = SecurityDomainTag("response_readable")
SYSTEM_PROMPT_TAG = SecurityDomainTag("system_prompt")
MODEL_IDENTITY_TAG = SecurityDomainTag("model_identity")


def make_controllable(name: str = "user_message", tag: SecurityDomainTag = USER_TAG) -> Controllable:
    return Controllable(name=name, security_domain=tag, value_type="text")


def make_observable(name: str = "response", tag: SecurityDomainTag = RESPONSE_TAG) -> Observable:
    return Observable(name=name, security_domain=tag)


def model_observable(model: str) -> ObservableValue:
    return ObservableValue(observable=make_observable("model", MODEL_IDENTITY_TAG), content=model)


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


def fake_llm() -> AsyncMock:
    return AsyncMock()
