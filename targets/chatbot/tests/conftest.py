"""Shared fixtures for ChatbotTarget tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from chatbot_target import (
    MODEL_IDENTITY_TAG,
    MODEL_TAG,
    SYSTEM_PROMPT_READABLE_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    USER_TAG,
    ChatbotTarget,
)


def make_litellm_response(content: str) -> MagicMock:
    """Build a mock litellm ModelResponse."""
    from litellm import ModelResponse

    resp = MagicMock(spec=ModelResponse)
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


@pytest.fixture
def target() -> ChatbotTarget:
    """A ChatbotTarget with test credentials."""
    return ChatbotTarget(model="test-model", api_key="sk-test")


@pytest.fixture
def target_with_base() -> ChatbotTarget:
    """A ChatbotTarget with api_base set."""
    return ChatbotTarget(model="test-model", api_key="sk-test", api_base="http://localhost:8080")
