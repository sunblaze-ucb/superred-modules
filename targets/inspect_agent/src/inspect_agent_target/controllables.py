"""Controllable singletons + builders exposed by :class:`InspectAgentTarget`.

Fixed injection points (module-level singletons):

- ``user_prompt`` (scope ``user``): override the user instruction the Task set.
  This is the jailbreak / prompt-attack surface.
- ``system_prompt`` (scope ``system_prompt``): override the agent system prompt.
- four ``tool_catalog_*`` controllables: edit the tool *registry* (which tools
  exist / their docs), fired once at run start.

Per-tool injection points (built dynamically by the target):

- one ``tool:<name>`` controllable per configured tool, scoped to that tool's
  trust boundary, fired as a ``ControllablePostCallEvent`` after the tool
  returns.  See :func:`tool_output_controllable`.

A passthrough optimizer (no injection) leaves everything at the Task-configured
values, which for a faithful benchmark port reproduces its baseline.
"""

from __future__ import annotations

from superred.core.types.controllable import Controllable
from superred.core.types.security_domain import SecurityDomainTag

from inspect_agent_target.security_tags import (
    SYSTEM_PROMPT_TAG,
    TOOL_CATALOGUE_ADDABLE_TAG,
    TOOL_CATALOGUE_TAG,
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
# Tool-catalogue controllables (the registry: which tools exist; fired once).
#
# The target ALWAYS fires these; the Controller's security_domain_filter
# decides whether a given optimizer (by scope) may actually inject.  Scopes:
# register is the weakest (TOOL_CATALOGUE_ADDABLE_TAG); replace/unregister/
# rewrite are broad (TOOL_CATALOGUE_TAG).  Initial tools are still configured
# statically by the Task.  Distinct from the per-tool ``tool:<name>`` output
# controllables below (the registry vs the content a tool returns).
# ---------------------------------------------------------------------------

TOOL_CATALOG_REGISTER_CTRL: Controllable = Controllable(
    name="tool_catalog_register",
    security_domain=TOOL_CATALOGUE_ADDABLE_TAG,
    description=(
        "Register a new attacker-defined tool with a fixed canned return.  "
        'JSON value: {"name": str, "description": str, '
        '"parameters_schema": dict (JSON-Schema object), "fake_return": Any}.'
    ),
    value_type="json",
)

TOOL_CATALOG_REPLACE_CTRL: Controllable = Controllable(
    name="tool_catalog_replace",
    security_domain=TOOL_CATALOGUE_TAG,
    description=(
        "Shadow an existing tool: when the agent calls the named tool, the "
        'attacker-supplied canned value is returned instead.  JSON value: '
        '{"name": str, "fake_return": Any, "description"?: str}.'
    ),
    value_type="json",
)

TOOL_CATALOG_UNREGISTER_CTRL: Controllable = Controllable(
    name="tool_catalog_unregister",
    security_domain=TOOL_CATALOGUE_TAG,
    description='Remove a tool from the catalogue.  JSON value: {"name": str}.',
    value_type="json",
)

TOOL_CATALOG_REWRITE_DOC_CTRL: Controllable = Controllable(
    name="tool_catalog_rewrite_doc",
    security_domain=TOOL_CATALOGUE_TAG,
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

# ---------------------------------------------------------------------------
# Per-tool output controllables (the indirect-prompt-injection surface).
#
# One Controllable per tool the agent can call, fired as a
# ControllablePostCallEvent after that tool returns, carrying the tool's
# legitimate return as ``answer``.  A ControllableInjection replaces what the
# agent sees, so an attacker can poison the data a specific tool returns.  Each
# is scoped to the tool's *trust boundary* (a claim-supplied tag under the
# ``tools`` root); tools sharing a boundary share a scope.  Post-call only: the
# tools are side-effect-free, so rewriting the whole return subsumes pre-call
# request tampering.
# ---------------------------------------------------------------------------


def tool_output_controllable(
    tool_name: str, security_domain: SecurityDomainTag
) -> Controllable:
    """Build the per-tool output-injection Controllable for *tool_name*.

    Named ``tool:<tool_name>`` and scoped to *security_domain* (the tool's trust
    boundary).  ``Controllable`` is a frozen, value-equal dataclass, so building
    this for the same ``(tool_name, security_domain)`` always yields an equal
    instance -- the target can rebuild it at fire time and the optimizer still
    matches the controllable it saw in ``get_controllables``.
    """
    return Controllable(
        name=f"tool:{tool_name}",
        security_domain=security_domain,
        description=(
            f"Replace the value the {tool_name!r} tool returns to the agent "
            "(indirect prompt injection). Fired once per call to this tool; the "
            "event's answer carries the legitimate output."
        ),
        value_type="text",
    )


CONTROLLABLES: list[Controllable] = [
    SYSTEM_PROMPT_CTRL,
    USER_PROMPT_CTRL,
    *TOOL_CATALOG_CTRLS,
]
"""The fixed Controllables the target always exposes, in stable order.  The
per-tool ``tool:<name>`` controllables are appended by the target, built from the
configured tool set + the claim's tool->tag map."""


__all__ = [
    "SYSTEM_PROMPT_CTRL",
    "USER_PROMPT_CTRL",
    "TOOL_CATALOG_REGISTER_CTRL",
    "TOOL_CATALOG_REPLACE_CTRL",
    "TOOL_CATALOG_UNREGISTER_CTRL",
    "TOOL_CATALOG_REWRITE_DOC_CTRL",
    "TOOL_CATALOG_CTRLS",
    "tool_output_controllable",
    "CONTROLLABLES",
]
