"""OpenClaw target module for superred."""

from openclaw_target.constants import (
    EXTERNAL_DATA_TAG,
    INTERNAL_CONTEXT_TAG,
    MODEL_TAG,
    OPENCLAW_DOMAIN,
    SYSTEM_TAG,
    TOOL_CATALOG_TAG,
    USER_INPUT_TAG,
)
from openclaw_target.target import (
    FILE_CONTENT_CTRL,
    MODEL_SYSTEM_PROMPT_CTRL,
    OpenClawTarget,
    USER_MESSAGE_CTRL,
    WEB_CONTENT_CTRL,
)

__all__ = [
    "OpenClawTarget",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    "EXTERNAL_DATA_TAG",
    "INTERNAL_CONTEXT_TAG",
    "TOOL_CATALOG_TAG",
    "MODEL_TAG",
    "OPENCLAW_DOMAIN",
    "USER_MESSAGE_CTRL",
    "WEB_CONTENT_CTRL",
    "FILE_CONTENT_CTRL",
    "MODEL_SYSTEM_PROMPT_CTRL",
]
