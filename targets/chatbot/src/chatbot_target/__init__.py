"""Chatbot target for superred.

Wraps any LLM accessible via litellm for chatbot evaluation.
Supports both single-turn and multi-turn conversations -- the
optimizer controls conversation length via controllable events.
"""

from chatbot_target.target import (
    BLOCKED_OBSERVABLE_NAME,
    BLOCKED_RESPONSE_MARKER,
    MODEL_IDENTITY_TAG,
    MODEL_TAG,
    RESPONSE_READABLE_TAG,
    SYSTEM_PROMPT_READABLE_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    USER_TAG,
    ChatbotTarget,
    content_filter_reason,
)

__all__ = [
    "ChatbotTarget",
    "SYSTEM_TAG",
    "SYSTEM_PROMPT_TAG",
    "SYSTEM_PROMPT_READABLE_TAG",
    "MODEL_TAG",
    "RESPONSE_READABLE_TAG",
    "MODEL_IDENTITY_TAG",
    "USER_TAG",
    "BLOCKED_RESPONSE_MARKER",
    "BLOCKED_OBSERVABLE_NAME",
    "content_filter_reason",
]
