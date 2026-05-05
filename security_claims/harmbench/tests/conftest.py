"""Shared fixtures for secclaim-harmbench tests.

All tests run without network access. Real LLM calls are forbidden;
the judge is mocked at construction or via :class:`AsyncMock`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from secclaim_harmbench.judge import HarmBenchJudge


def _make_judge() -> HarmBenchJudge:
    """Construct a HarmBenchJudge with non-network args (real .judge() never called)."""
    return HarmBenchJudge(
        model="openai/test-stub",
        api_base="https://invalid.local",
        api_key="sk-test",
    )


@pytest.fixture
def stub_judge_yes() -> HarmBenchJudge:
    """A HarmBenchJudge whose .judge() always returns (True, '<stub: Yes>')."""
    judge = _make_judge()
    judge.judge = AsyncMock(return_value=(True, "<stub: Yes>"))  # type: ignore[method-assign]
    return judge


@pytest.fixture
def stub_judge_no() -> HarmBenchJudge:
    """A HarmBenchJudge whose .judge() always returns (False, '<stub: No>')."""
    judge = _make_judge()
    judge.judge = AsyncMock(return_value=(False, "<stub: No>"))  # type: ignore[method-assign]
    return judge


def fake_row(
    *,
    behavior_id: str = "stub_id",
    behavior: str = "Stub behavior",
    functional: str = "standard",
    semantic: str = "harmful",
    tags: str = "",
    context: str = "",
    split: str = "test",
) -> dict[str, str]:
    """Construct a single row dict with all required keys."""
    return {
        "Behavior": behavior,
        "FunctionalCategory": functional,
        "SemanticCategory": semantic,
        "Tags": tags,
        "ContextString": context,
        "BehaviorID": behavior_id,
        "_split": split,
    }


class StubChatbotTarget:
    """Lightweight ChatbotTarget stub for unit-testing Task lifecycle.

    Records calls to ``set_config`` and replays a fixed ``last_response``.
    Only ``set_config`` and ``query`` are exercised by HarmBenchTask
    on the evaluation path; the full ChatbotTarget API (config_specs,
    query_specs, controllables, run, etc.) is not needed in unit tests.
    """

    def __init__(self, *, last_response: str = "") -> None:
        self.last_response = last_response
        self.config_calls: list[tuple[str, str]] = []

    def set_config(self, name: str, value: str) -> None:
        self.config_calls.append((name, value))

    def query(self, name: str, **params: str) -> str:
        if name == "last_response":
            return self.last_response
        return ""


@pytest.fixture
def stub_target() -> StubChatbotTarget:
    return StubChatbotTarget()
