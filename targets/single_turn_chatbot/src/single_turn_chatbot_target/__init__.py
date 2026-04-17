"""Chatbot target for superred.

Wraps any LLM accessible via litellm for chatbot evaluation.
Supports both single-turn and multi-turn conversations — the
optimizer controls conversation length via controllable events.
"""

from single_turn_chatbot_target.target import (
    RESPONSE_TAG,
    SYSTEM_PROMPT_READABLE_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    USER_TAG,
    ChatbotTarget,
)

# Backwards-compatible alias
SingleTurnChatbotTarget = ChatbotTarget

__all__ = [
    "ChatbotTarget",
    "SingleTurnChatbotTarget",
    "SYSTEM_TAG",
    "SYSTEM_PROMPT_TAG",
    "SYSTEM_PROMPT_READABLE_TAG",
    "RESPONSE_TAG",
    "USER_TAG",
]
