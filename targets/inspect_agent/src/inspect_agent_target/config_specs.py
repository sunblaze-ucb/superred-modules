"""ConfigSpec declarations for :class:`InspectAgentTarget`.

All slots are set by the Task in ``configure_target`` (string-only, per the
framework's ``set_config(name: str, value: str)`` contract):

- ``system_prompt`` - agent system prompt (empty string = no system message).
- ``user_prompt`` - the user instruction the agent acts on.  An attacker
  with ``user`` in scope can override it via the ``user_prompt`` Controllable.
- ``tool_names`` - JSON-encoded ``list[str]`` of tool names to expose this
  run.  Each name is resolved to an inspect ``Tool`` via the resolver the
  target was constructed with.  ``""`` or ``"[]"`` = no tools.
- ``tool_choice`` - one of ``"auto"``, ``"any"``, ``"none"`` (default ``"auto"``).
- ``message_limit`` - integer (as a string) cap on total messages in the
  rollout.  Empty = the target's construction default.
- ``model`` - optional per-run litellm-style model id override.
"""

from __future__ import annotations

from superred.core.types.state import ConfigSpec

from inspect_agent_target.security_tags import (
    MODEL_IDENTITY_TAG,
    SYSTEM_PROMPT_TAG,
    TOOLS_TAG,
    USER_TAG,
)

SYSTEM_PROMPT_SPEC: ConfigSpec = ConfigSpec(
    name="system_prompt",
    security_domain=SYSTEM_PROMPT_TAG,
    description=(
        "Agent system prompt.  Set by the Task in configure_target.  "
        "Empty string means no system message is prepended."
    ),
)

USER_PROMPT_SPEC: ConfigSpec = ConfigSpec(
    name="user_prompt",
    security_domain=USER_TAG,
    description=(
        "The user instruction the agent acts on.  The Task sets the benign "
        "(or canonical) prompt; an attacker with the user tag in scope can "
        "override it via the user_prompt Controllable at run start."
    ),
)

TOOL_NAMES_SPEC: ConfigSpec = ConfigSpec(
    name="tool_names",
    security_domain=TOOLS_TAG,
    description=(
        "JSON-encoded list[str] of tool names to expose for this run.  Each "
        "name is resolved to an inspect Tool via the resolver the target was "
        "constructed with.  Empty string or '[]' means no tools."
    ),
)

TOOL_CHOICE_SPEC: ConfigSpec = ConfigSpec(
    name="tool_choice",
    security_domain=TOOLS_TAG,
    description="Tool choice: one of 'auto', 'any', 'none' (default 'auto').",
)

MESSAGE_LIMIT_SPEC: ConfigSpec = ConfigSpec(
    name="message_limit",
    security_domain=TOOLS_TAG,
    description=(
        "Integer (as string) cap on total messages in the rollout.  Empty "
        "string uses the target's construction default."
    ),
)

MODEL_SPEC: ConfigSpec = ConfigSpec(
    name="model",
    security_domain=MODEL_IDENTITY_TAG,
    description=(
        "Optional per-run litellm-style model id override, e.g. "
        "'openai/gpt-4o-2024-08-06'.  Empty string keeps the construction model."
    ),
)


CONFIG_SPECS: list[ConfigSpec] = [
    SYSTEM_PROMPT_SPEC,
    USER_PROMPT_SPEC,
    TOOL_NAMES_SPEC,
    TOOL_CHOICE_SPEC,
    MESSAGE_LIMIT_SPEC,
    MODEL_SPEC,
]
"""All ConfigSpecs returned from :attr:`InspectAgentTarget.config_specs`."""

CONFIG_SPEC_NAMES: frozenset[str] = frozenset(s.name for s in CONFIG_SPECS)
"""Names of every config slot, for O(1) validation in ``set_config``."""


__all__ = [
    "SYSTEM_PROMPT_SPEC",
    "USER_PROMPT_SPEC",
    "TOOL_NAMES_SPEC",
    "TOOL_CHOICE_SPEC",
    "MESSAGE_LIMIT_SPEC",
    "MODEL_SPEC",
    "CONFIG_SPECS",
    "CONFIG_SPEC_NAMES",
]
