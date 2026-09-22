"""Shared fixtures/helpers for the aegis_claim test suite.

All tests are offline: the dataset is vendored, and every judge LLM call is
mocked. No test hits a network or a real model.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from superred.core.llm import LLMClient
from superred.core.types.llm import LLMUsage


def fake_response(content: str) -> MagicMock:
    """A mock litellm ``ModelResponse`` whose first choice carries ``content``."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def fake_llm(content: str, *, calls: int = 1, cost: float = 0.01) -> AsyncMock:
    """An ``AsyncMock(spec=LLMClient)`` whose ``complete()`` returns ``content``."""
    fake = AsyncMock(spec=LLMClient)
    fake.complete.return_value = fake_response(content)
    fake.usage = LLMUsage(calls=calls, cost=cost)
    return fake


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record judge back-off delays instead of waiting for them."""
    delays: list[float] = []

    async def _record(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("aegis_claim.judge_failure._default_sleep", _record)
    return delays
