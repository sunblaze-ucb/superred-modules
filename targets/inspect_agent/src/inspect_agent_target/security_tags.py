"""SecurityDomain forest for the general inspect-agent target.

Structured in the AgentDojo style: three independent root trees, with
capability subsumption encoded by the parent/child hierarchy (a scope holding a
parent tag includes all its descendants, so the Controller can scope broadly or
narrowly).

- ``system``: agent-side surfaces (everything the agent IS / how it is
  configured / its own trace).  An attacker with ``system`` holds every
  agent-side capability below it.
    - ``system_prompt`` (writable) -> ``system_prompt_readable`` (read-only).
    - ``tool_catalogue`` (broad registry write: set tools / replace / unregister
      / rewrite-description) -> ``tool_catalogue_readable`` (read the listing) and
      ``tool_catalogue_addable`` (register-only, the weakest write).
    - ``model_identity``: which model powers the agent.
    - ``agent_trace`` (read the run trace) -> ``agent_trace_messages``,
      ``agent_trace_tool_calls``, ``agent_trace_tool_responses``.
- ``user``: the user-prompt channel (the jailbreak / prompt-attack surface).
- ``tool_output``: the content tools return to the agent.  Injecting here
  replaces a tool's return value before the agent sees it -- the indirect-
  prompt-injection surface (poisoning data the agent reads).  This is the
  analogue of AgentDojo's tool-content surface; the *read* side lives under
  ``agent_trace.agent_trace_tool_responses``.

Each tag is a module-level singleton so ``scope_includes`` compares by identity.
"""

from __future__ import annotations

from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag

# ===========================================================================
# Tree 1: system (agent-side surfaces)
# ===========================================================================

SYSTEM_TAG: SecurityDomainTag = SecurityDomainTag("system")
"""Root of the agent-side tree.  Holds every agent-side capability below it."""

# --- system prompt ---------------------------------------------------------

SYSTEM_PROMPT_TAG: SecurityDomainTag = SecurityDomainTag(
    "system_prompt", parent=SYSTEM_TAG,
)
"""Writable agent system prompt.  Implies read via :data:`SYSTEM_PROMPT_READABLE_TAG`."""

SYSTEM_PROMPT_READABLE_TAG: SecurityDomainTag = SecurityDomainTag(
    "system_prompt_readable", parent=SYSTEM_PROMPT_TAG,
)
"""Read-only view of the current system prompt."""

# --- tool catalogue (the tool registry) ------------------------------------

TOOL_CATALOGUE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue", parent=SYSTEM_TAG,
)
"""Broad tool-registry write: set the initial tool set and perform any catalogue
edit (replace / unregister / rewrite-description).  Implies the read-only
:data:`TOOL_CATALOGUE_READABLE_TAG` and the register-only
:data:`TOOL_CATALOGUE_ADDABLE_TAG`."""

TOOL_CATALOGUE_READABLE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_readable", parent=TOOL_CATALOGUE_TAG,
)
"""Read-only view of the current tool catalogue (the catalogue-listing observable)."""

TOOL_CATALOGUE_ADDABLE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_addable", parent=TOOL_CATALOGUE_TAG,
)
"""Register-only catalogue write capability: the weakest write.  An attacker can
add new (attacker-defined) tools but cannot replace, unregister, or rewrite
existing ones.  Models a malicious-MCP that only contributes extra tools."""

# --- model identity --------------------------------------------------------

MODEL_IDENTITY_TAG: SecurityDomainTag = SecurityDomainTag(
    "model_identity", parent=SYSTEM_TAG,
)
"""Knowledge of which model the agent uses."""

# --- agent trace (read access) ---------------------------------------------

AGENT_TRACE_TAG: SecurityDomainTag = SecurityDomainTag("agent_trace", parent=SYSTEM_TAG)
"""Aggregate read access to the agent's run trace.  Implies the children."""

AGENT_TRACE_MESSAGES_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_messages", parent=AGENT_TRACE_TAG,
)
"""Read access to the agent's chat-message stream."""

AGENT_TRACE_TOOL_CALLS_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_tool_calls", parent=AGENT_TRACE_TAG,
)
"""Read access to the tool calls the agent emits."""

AGENT_TRACE_TOOL_RESPONSES_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_tool_responses", parent=AGENT_TRACE_TAG,
)
"""Read access to the tool return values the agent observes (after any
tool-output injection has been applied)."""

# ===========================================================================
# Tree 2: user (the user-prompt channel)
# ===========================================================================

USER_TAG: SecurityDomainTag = SecurityDomainTag("user")
"""User-prompt channel.  An attacker with ``user`` can override the benign user
prompt set by the Task (jailbreak surface)."""

# ===========================================================================
# Tree 3: tool_output (content tools return to the agent)
# ===========================================================================

TOOL_OUTPUT_TAG: SecurityDomainTag = SecurityDomainTag("tool_output")
"""The content tools return to the agent.  An attacker with this replaces a
tool's return value before the agent sees it -- indirect prompt injection.
Distinct from ``tool_catalogue`` (the registry: which tools exist): this is the
*content* the tools return.  The read side is ``agent_trace_tool_responses``."""

# ===========================================================================
# Assembled forest
# ===========================================================================

DOMAIN: SecurityDomain = SecurityDomain([
    # system tree
    SYSTEM_TAG,
    SYSTEM_PROMPT_TAG, SYSTEM_PROMPT_READABLE_TAG,
    TOOL_CATALOGUE_TAG, TOOL_CATALOGUE_READABLE_TAG, TOOL_CATALOGUE_ADDABLE_TAG,
    MODEL_IDENTITY_TAG,
    AGENT_TRACE_TAG,
    AGENT_TRACE_MESSAGES_TAG, AGENT_TRACE_TOOL_CALLS_TAG, AGENT_TRACE_TOOL_RESPONSES_TAG,
    # user tree
    USER_TAG,
    # tool-output tree
    TOOL_OUTPUT_TAG,
])
"""The full security-domain forest exposed by :class:`InspectAgentTarget`."""


__all__ = [
    "SYSTEM_TAG",
    "SYSTEM_PROMPT_TAG",
    "SYSTEM_PROMPT_READABLE_TAG",
    "TOOL_CATALOGUE_TAG",
    "TOOL_CATALOGUE_READABLE_TAG",
    "TOOL_CATALOGUE_ADDABLE_TAG",
    "MODEL_IDENTITY_TAG",
    "AGENT_TRACE_TAG",
    "AGENT_TRACE_MESSAGES_TAG",
    "AGENT_TRACE_TOOL_CALLS_TAG",
    "AGENT_TRACE_TOOL_RESPONSES_TAG",
    "USER_TAG",
    "TOOL_OUTPUT_TAG",
    "DOMAIN",
]
