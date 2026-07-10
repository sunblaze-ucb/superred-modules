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
from openclaw_target.factory import openclaw_target_factory
from openclaw_target.docker_runtime import DEFAULT_DOCKER_IMAGE
from openclaw_target.target import (
    ALLOWED_WORKSPACE_BOOTSTRAP_FILES,
    FILE_CONTENT_CTRL,
    MEMORY_POISON_CTRL,
    MESSAGE_CONTENT_CTRL,
    MODEL_RESPONSE_CTRL,
    MODEL_SYSTEM_PROMPT_CTRL,
    SHELL_OUTPUT_CTRL,
    TOOL_OUTPUT_CONTROLLABLES,
    OpenClawTarget,
    USER_MESSAGE_CTRL,
    WEB_CONTENT_CTRL,
)

__all__ = [
    "OpenClawTarget",
    "openclaw_target_factory",
    "DEFAULT_DOCKER_IMAGE",
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
    "SHELL_OUTPUT_CTRL",
    "MESSAGE_CONTENT_CTRL",
    "MEMORY_POISON_CTRL",
    "MODEL_SYSTEM_PROMPT_CTRL",
    "MODEL_RESPONSE_CTRL",
    "TOOL_OUTPUT_CONTROLLABLES",
    "ALLOWED_WORKSPACE_BOOTSTRAP_FILES",
]
