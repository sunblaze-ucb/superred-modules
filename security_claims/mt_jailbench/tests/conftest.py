"""Shared fixtures for the mt_jailbench test suite (all offline).

The judge fixture patches ``mt_jailbench_claim.judge.acompletion`` so tests run
without a network. Judge answers supplied by tests are synthetic neutral
strings (``[[1]]``, ``{"score": 5}``, ``unsafe``) -- never real rubric or
dataset content.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from mt_jailbench_claim.judge import JudgeConfig


@dataclass
class _MockMessage:
    content: str | None


@dataclass
class _MockChoice:
    message: _MockMessage


@dataclass
class _MockResponse:
    choices: list[_MockChoice]


def make_response(content: str | None) -> _MockResponse:
    return _MockResponse(choices=[_MockChoice(message=_MockMessage(content=content))])


@pytest.fixture
def mock_judge(monkeypatch: pytest.MonkeyPatch) -> Callable[[Sequence[str]], AsyncMock]:
    """Return a setter that scripts successive ``acompletion`` answers in order."""

    mock = AsyncMock()
    monkeypatch.setattr("mt_jailbench_claim.judge.acompletion", mock)

    def _set(contents: Sequence[str]) -> AsyncMock:
        mock.side_effect = [make_response(c) for c in contents]
        return mock

    return _set


@pytest.fixture
def mock_judge_raises(monkeypatch: pytest.MonkeyPatch) -> Callable[[BaseException], AsyncMock]:
    """Make ``acompletion`` raise the supplied exception on every call."""

    mock = AsyncMock()
    monkeypatch.setattr("mt_jailbench_claim.judge.acompletion", mock)

    def _set(exc: BaseException) -> AsyncMock:
        mock.side_effect = exc
        return mock

    return _set


@pytest.fixture
def judge_config() -> JudgeConfig:
    return JudgeConfig(api_base="https://example.invalid/v1", api_key="test-key", max_retries=1)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def _record(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("mt_jailbench_claim.judge_failure._default_sleep", _record)
    return delays


class FakeChatbotTarget:
    """Minimal stand-in exposing the ChatbotTarget contract the task uses."""

    def __init__(self, last_response: str) -> None:
        self._last_response = last_response
        self.config: dict[str, str] = {}

    def set_config(self, name: str, value: str) -> None:
        self.config[name] = value

    def query(self, name: str, **_params: Any) -> str:
        if name == "last_response":
            return self._last_response
        raise KeyError(name)
