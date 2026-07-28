"""Shared fixtures for ChatbotTarget tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from chatbot_target import (
    MODEL_IDENTITY_TAG,
    MODEL_TAG,
    RESPONSE_READABLE_TAG,
    SYSTEM_PROMPT_READABLE_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    USER_TAG,
    ChatbotTarget,
)


def make_litellm_response(
    content: str | None,
    *,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
    choices: bool = True,
) -> MagicMock:
    """Build a mock litellm ModelResponse.

    ``finish_reason`` carries the in-band form of a content-filter block, so it
    has to be settable; ``choices=False`` builds the shape a provider returns
    when it produced nothing at all.
    """
    from litellm import ModelResponse

    resp = MagicMock(spec=ModelResponse)
    if not choices:
        resp.choices = []
        return resp
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.reasoning_content = reasoning_content
    resp.choices[0].finish_reason = finish_reason
    return resp


@pytest.fixture
def target() -> ChatbotTarget:
    """A ChatbotTarget with test credentials."""
    return ChatbotTarget(model="test-model", api_key="sk-test")


@pytest.fixture
def fast_target() -> ChatbotTarget:
    """A ChatbotTarget whose retry backoff does not sleep, for retry-path tests."""
    return ChatbotTarget(model="test-model", api_key="sk-test", retry_backoff_s=0.0)


@pytest.fixture
def target_with_base() -> ChatbotTarget:
    """A ChatbotTarget with api_base set."""
    return ChatbotTarget(model="test-model", api_key="sk-test", api_base="http://localhost:8080")
