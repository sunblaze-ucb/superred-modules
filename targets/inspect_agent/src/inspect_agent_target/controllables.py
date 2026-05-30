"""Controllable singletons exposed by :class:`InspectAgentTarget`.

The generic agent target exposes two injection points, fired as
``ControllablePreCallEvent``s at run start:

- ``user_prompt`` (scope ``user``): override the user instruction the Task
  set.  This is the jailbreak / prompt-attack surface.
- ``system_prompt`` (scope ``system_prompt``): override the agent system
  prompt the Task set.

A passthrough optimizer (no injection) leaves both at the Task-configured
values, which for a faithful benchmark port reproduces its baseline.
"""

from __future__ import annotations

from superred.core.types.controllable import Controllable

from inspect_agent_target.security_tags import (
    SYSTEM_PROMPT_TAG,
    TOOLS_ADDABLE_TAG,
    TOOLS_TAG,
    USER_TAG,
)

SYSTEM_PROMPT_CTRL: Controllable = Controllable(
    name="system_prompt",
    security_domain=SYSTEM_PROMPT_TAG,
    description="Override the agent's system prompt for this run.",
    value_type="text",
)

USER_PROMPT_CTRL: Controllable = Controllable(
    name="user_prompt",
    security_domain=USER_TAG,
    description="Override the user instruction set by the Task (jailbreak surface).",
    value_type="text",
)

# ---------------------------------------------------------------------------
# Tool-catalogue controllables (fired before every LLM turn).
#
# The target ALWAYS fires these; the Controller's security_domain_filter
# decides whether a given optimizer (by scope) may actually inject.  Scopes:
# register is the weakest (TOOLS_ADDABLE_TAG); replace/unregister/rewrite are
# broad (TOOLS_TAG).  Initial tools are still configured statically by the Task.
# ---------------------------------------------------------------------------

TOOL_CATALOG_REGISTER_CTRL: Controllable = Controllable(
    name="tool_catalog_register",
    security_domain=TOOLS_ADDABLE_TAG,
    description=(
        "Register a new attacker-defined tool with a fixed canned return.  "
        'JSON value: {"name": str, "description": str, '
        '"parameters_schema": dict (JSON-Schema object), "fake_return": Any}.'
    ),
    value_type="json",
)

TOOL_CATALOG_REPLACE_CTRL: Controllable = Controllable(
    name="tool_catalog_replace",
    security_domain=TOOLS_TAG,
    description=(
        "Shadow an existing tool: when the agent calls the named tool, the "
        'attacker-supplied canned value is returned instead.  JSON value: '
        '{"name": str, "fake_return": Any, "description"?: str}.'
    ),
    value_type="json",
)

TOOL_CATALOG_UNREGISTER_CTRL: Controllable = Controllable(
    name="tool_catalog_unregister",
    security_domain=TOOLS_TAG,
    description='Remove a tool from the catalogue.  JSON value: {"name": str}.',
    value_type="json",
)

TOOL_CATALOG_REWRITE_DOC_CTRL: Controllable = Controllable(
    name="tool_catalog_rewrite_doc",
    security_domain=TOOLS_TAG,
    description=(
        "Rewrite an existing tool's description without changing its behaviour.  "
        'JSON value: {"name": str, "description": str}.'
    ),
    value_type="json",
)

TOOL_CATALOG_CTRLS: tuple[Controllable, ...] = (
    TOOL_CATALOG_REGISTER_CTRL,
    TOOL_CATALOG_REPLACE_CTRL,
    TOOL_CATALOG_UNREGISTER_CTRL,
    TOOL_CATALOG_REWRITE_DOC_CTRL,
)

CONTROLLABLES: list[Controllable] = [
    SYSTEM_PROMPT_CTRL,
    USER_PROMPT_CTRL,
    *TOOL_CATALOG_CTRLS,
]
"""Every Controllable the target exposes, in stable order."""


__all__ = [
    "SYSTEM_PROMPT_CTRL",
    "USER_PROMPT_CTRL",
    "TOOL_CATALOG_REGISTER_CTRL",
    "TOOL_CATALOG_REPLACE_CTRL",
    "TOOL_CATALOG_UNREGISTER_CTRL",
    "TOOL_CATALOG_REWRITE_DOC_CTRL",
    "TOOL_CATALOG_CTRLS",
    "CONTROLLABLES",
]
