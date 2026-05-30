"""SecurityDomain forest for the general inspect-agent target.

This target is benchmark-agnostic, so the forest is deliberately small and
covers only the surfaces a generic tool-calling agent exposes:

- ``user``: the user-prompt channel.  An attacker with ``user`` can
  override the benign user prompt set by the Task (e.g. apply a jailbreak)
  via the ``user_prompt`` Controllable.
- ``system_prompt``: the agent system prompt (writable), with a read-only
  child for the system-prompt observable.
- ``model_identity``: knowledge of which model powers the agent.
- ``tools``: the agent's tool catalogue (which tools are exposed + the
  tool-execution knobs).
- ``agent_trace``: read access to the agent's run trace, with children for
  the chat-message stream and the tool-call stream.

A SecurityClaim built on this target pins its Scores to whichever tag
matches its threat model (AgentHarm pins to ``user``).  Each tag is a
module-level singleton so ``scope_includes`` compares by identity.
"""

from __future__ import annotations

from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag

# --- user channel ----------------------------------------------------------

USER_TAG: SecurityDomainTag = SecurityDomainTag("user")
"""User-prompt channel.  An attacker with ``user`` can override the benign
user prompt set by the Task (jailbreak surface)."""

# --- system prompt ---------------------------------------------------------

SYSTEM_PROMPT_TAG: SecurityDomainTag = SecurityDomainTag("system_prompt")
"""Writable agent system prompt.  Implies read access via
:data:`SYSTEM_PROMPT_READABLE_TAG`."""

SYSTEM_PROMPT_READABLE_TAG: SecurityDomainTag = SecurityDomainTag(
    "system_prompt_readable", parent=SYSTEM_PROMPT_TAG,
)
"""Read-only view of the current system prompt."""

# --- model identity --------------------------------------------------------

MODEL_IDENTITY_TAG: SecurityDomainTag = SecurityDomainTag("model_identity")
"""Knowledge of which model the agent uses."""

# --- tool catalogue --------------------------------------------------------

TOOLS_TAG: SecurityDomainTag = SecurityDomainTag("tools")
"""Broad tool-catalogue capability: set the initial tool set and perform any
catalogue edit (replace / unregister / rewrite-description).  Implies the
weaker register-only :data:`TOOLS_ADDABLE_TAG` and the read-only
:data:`TOOLS_READABLE_TAG`."""

TOOLS_READABLE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tools_readable", parent=TOOLS_TAG,
)
"""Read-only view of the current tool catalogue (the catalog-listing observable)."""

TOOLS_ADDABLE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tools_addable", parent=TOOLS_TAG,
)
"""Register-only catalogue write capability: the weakest write.  An attacker
can add new (attacker-defined) tools but cannot replace, unregister, or rewrite
existing ones.  Models a malicious-MCP that only contributes extra tools."""

# --- agent trace -----------------------------------------------------------

AGENT_TRACE_TAG: SecurityDomainTag = SecurityDomainTag("agent_trace")
"""Aggregate read access to the agent's run trace.  Implies the children."""

AGENT_TRACE_MESSAGES_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_messages", parent=AGENT_TRACE_TAG,
)
"""Read access to the agent's chat-message stream."""

AGENT_TRACE_TOOL_CALLS_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_tool_calls", parent=AGENT_TRACE_TAG,
)
"""Read access to the tool calls the agent emits."""

# --- assembled forest ------------------------------------------------------

DOMAIN: SecurityDomain = SecurityDomain([
    USER_TAG,
    SYSTEM_PROMPT_TAG, SYSTEM_PROMPT_READABLE_TAG,
    MODEL_IDENTITY_TAG,
    TOOLS_TAG, TOOLS_READABLE_TAG, TOOLS_ADDABLE_TAG,
    AGENT_TRACE_TAG, AGENT_TRACE_MESSAGES_TAG, AGENT_TRACE_TOOL_CALLS_TAG,
])
"""The full security-domain forest exposed by :class:`InspectAgentTarget`."""


__all__ = [
    "USER_TAG",
    "SYSTEM_PROMPT_TAG",
    "SYSTEM_PROMPT_READABLE_TAG",
    "MODEL_IDENTITY_TAG",
    "TOOLS_TAG",
    "TOOLS_READABLE_TAG",
    "TOOLS_ADDABLE_TAG",
    "AGENT_TRACE_TAG",
    "AGENT_TRACE_MESSAGES_TAG",
    "AGENT_TRACE_TOOL_CALLS_TAG",
    "DOMAIN",
]
