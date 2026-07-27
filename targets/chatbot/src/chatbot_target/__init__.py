"""Chatbot target for superred.

Wraps any LLM accessible via litellm for chatbot evaluation.
Supports both single-turn and multi-turn conversations -- the
optimizer controls conversation length via controllable events.
"""

from chatbot_target.target import (
    MODEL_IDENTITY_TAG,
    MODEL_TAG,
    SYSTEM_PROMPT_READABLE_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    USER_TAG,
    ChatbotTarget,
)

__all__ = [
    "ChatbotTarget",
    "SYSTEM_TAG",
    "SYSTEM_PROMPT_TAG",
    "SYSTEM_PROMPT_READABLE_TAG",
    "MODEL_TAG",
    "MODEL_IDENTITY_TAG",
    "USER_TAG",
]
