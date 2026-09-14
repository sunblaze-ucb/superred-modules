"""safeclawarena_target: a superred Target over SafeClawArena's container harness.

Runs one SafeClawArena adversarial task against a containerized platform replica
(OpenClaw / NemoClaw), provisioning the task environment through the
vendored ``reset_env.sh``, executing the sessions, and capturing the post-run
state the paired ``safeclawarena_claim`` judge consumes. Pairs with the
``safeclawarena-claim`` package.
"""

from __future__ import annotations

from safeclawarena_target.factory import safeclawarena_target_factory
from safeclawarena_target.security_tags import (
    CONFIG_TAG,
    CRON_TAG,
    DIMENSION_TAG,
    DOMAIN,
    LEAK_CHANNEL_TAG,
    MEMORY_TAG,
    MODEL_IDENTITY_TAG,
    PERSISTENT_STATE_TAG,
    PLUGINS_TAG,
    RESPONSE_TAG,
    SIM_GOOGLE_SERVICE_TAGS,
    SIM_GOOGLE_SERVICES,
    SIM_GOOGLE_TAG,
    SKILLS_TAG,
    SUPPLY_CHAIN_TAG,
    SYSTEM_TAG,
    TASK_METADATA_TAG,
    TOOLS_TAG,
    USER_TAG,
    WORKSPACE_FILES_TAG,
    sim_google_call_tag,
)
from safeclawarena_target.target import (
    USER_MESSAGE_CTRL,
    WORKSPACE_FILE_CTRL,
    SafeClawArenaTarget,
    post_run_observables,
)

__all__ = [
    "SafeClawArenaTarget",
    "safeclawarena_target_factory",
    "post_run_observables",
    "DOMAIN",
    "DIMENSION_TAG",
    "LEAK_CHANNEL_TAG",
    "SYSTEM_TAG",
    "SUPPLY_CHAIN_TAG",
    "SKILLS_TAG",
    "PLUGINS_TAG",
    "PERSISTENT_STATE_TAG",
    "MEMORY_TAG",
    "CONFIG_TAG",
    "CRON_TAG",
    "RESPONSE_TAG",
    "MODEL_IDENTITY_TAG",
    "TASK_METADATA_TAG",
    "USER_TAG",
    "TOOLS_TAG",
    "WORKSPACE_FILES_TAG",
    "SIM_GOOGLE_TAG",
    "SIM_GOOGLE_SERVICES",
    "SIM_GOOGLE_SERVICE_TAGS",
    "sim_google_call_tag",
    "USER_MESSAGE_CTRL",
    "WORKSPACE_FILE_CTRL",
]
