"""safeclawarena_target: a superred Target over SafeClawArena's container harness.

Runs one SafeClawArena adversarial task against a containerized platform replica
(OpenClaw / NemoClaw / SecLaw), provisioning the task environment through the
vendored ``reset_env.sh``, executing the sessions, and capturing the post-run
state the paired ``safeclawarena_claim`` judge consumes. Pairs with the
``safeclawarena-claim`` package.
"""

from __future__ import annotations

from safeclawarena_target.factory import safeclawarena_target_factory
from safeclawarena_target.security_tags import (
    CONFIG_TAG,
    DIMENSION_TAG,
    DOMAIN,
    EXTERNAL_DATA_TAG,
    FILE_CONTENT_TAG,
    MEMORY_TAG,
    SIM_GOOGLE_TAG,
    SKILLS_TAG,
    SYSTEM_TAG,
    TOOLS_TAG,
    USER_TAG,
)
from safeclawarena_target.target import (
    EXTERNAL_DATA_CTRL,
    SafeClawArenaTarget,
    USER_MESSAGE_CTRL,
)

__all__ = [
    "SafeClawArenaTarget",
    "safeclawarena_target_factory",
    "DOMAIN",
    "DIMENSION_TAG",
    "SYSTEM_TAG",
    "SKILLS_TAG",
    "MEMORY_TAG",
    "CONFIG_TAG",
    "USER_TAG",
    "EXTERNAL_DATA_TAG",
    "SIM_GOOGLE_TAG",
    "FILE_CONTENT_TAG",
    "TOOLS_TAG",
    "USER_MESSAGE_CTRL",
    "EXTERNAL_DATA_CTRL",
]
