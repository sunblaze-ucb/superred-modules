"""Chatbot target for superred.

Wraps any LLM accessible via litellm for chatbot evaluation.
Supports both single-turn and multi-turn conversations -- the
optimizer controls conversation length via controllable events.
"""

from chatbot_target.target import (
    BLOCKED_OBSERVABLE_NAME,
    BLOCKED_RESPONSE_MARKER,
    EMPTY_OBSERVABLE_NAME,
    INVALID_INJECTION_OBSERVABLE_NAME,
    MISSING_OBSERVABLE_NAME,
    MODEL_IDENTITY_TAG,
    MODEL_TAG,
    OUTCOME_BLOCKED,
    OUTCOME_EMPTY,
    OUTCOME_INVALID_INJECTION,
    OUTCOME_NO_CONTENT,
    OUTCOME_OK,
    RESPONSE_READABLE_TAG,
    RETRY_OBSERVABLE_NAME,
    SYSTEM_PROMPT_READABLE_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    USER_TAG,
    ChatbotTarget,
    content_filter_reason,
    invalid_injection_reason,
    no_content_reason,
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
    "OUTCOME_OK",
    "OUTCOME_EMPTY",
    "OUTCOME_BLOCKED",
    "OUTCOME_NO_CONTENT",
    "OUTCOME_INVALID_INJECTION",
    "BLOCKED_RESPONSE_MARKER",
    "BLOCKED_OBSERVABLE_NAME",
    "EMPTY_OBSERVABLE_NAME",
    "MISSING_OBSERVABLE_NAME",
    "RETRY_OBSERVABLE_NAME",
    "INVALID_INJECTION_OBSERVABLE_NAME",
    "content_filter_reason",
    "no_content_reason",
    "invalid_injection_reason",
]
