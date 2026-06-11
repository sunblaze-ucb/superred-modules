"""SecurityDomain forest and tag instances for the AgentDojo composite target.

The forest has three independent root trees:

- ``system``: agent-side capabilities the attacker may control.  Subtree
  layout encodes capability subsumption: an attacker holding
  ``tool_catalogue`` automatically holds the weaker
  ``tool_catalogue_addable``.  Read-only access to a surface is not a
  separate tag; it is granted per threat model by listing the tag in the
  Controller's ``read_only`` set rather than its read & write ``scope``.
- ``user``: the user prompt channel.  A single tag (no children) because
  the user channel has no internal hierarchy in this threat model.
- ``tools``: a 2x2 grid classifying every readable data store by who
  authored the content and who provides the storage.  No tools-root
  hierarchy is enforced; the four leaves are siblings under ``tools``.

Tag names match the brief's specification.  Each tag is a module-level
singleton so callers can reference them by import and the same instance
is used everywhere ``scope_includes`` compares identity.

See ``ASSUMPTIONS.md`` Section C for the mapping rationale and the
per-tool quadrant assignment.
"""

from __future__ import annotations

from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag

# ---------------------------------------------------------------------------
# Tree 1: system
# ---------------------------------------------------------------------------

SYSTEM_TAG: SecurityDomainTag = SecurityDomainTag("system")
"""Root of the system tree.  An attacker with ``system`` holds every
system-side capability below it."""

PROMPT_TAG: SecurityDomainTag = SecurityDomainTag("prompt", parent=SYSTEM_TAG)
"""System prompt surface.  When ``prompt`` is in the Controller's read &
write ``scope`` the attacker may override it; listing it under
``read_only`` instead grants see-but-not-change access."""

TOOL_CATALOGUE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue", parent=SYSTEM_TAG,
)
"""Broadest tool-catalogue capability.  Implies replace, unregister,
rewrite-description, and the weaker :data:`TOOL_CATALOGUE_ADDABLE_TAG`
(register-only).  The catalogue-listing observable carries this tag, so
listing ``tool_catalogue`` under ``read_only`` (rather than ``scope``)
grants the listing without any edit capability."""

TOOL_CATALOGUE_ADDABLE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_addable", parent=TOOL_CATALOGUE_TAG,
)
"""Register-only catalogue write capability.  Weakest write capability:
the attacker can add new tools but cannot replace, unregister, or rewrite
existing entries.  Models a malicious-MCP that can only contribute
additional tools, not shadow existing ones."""

MODEL_IDENTITY_TAG: SecurityDomainTag = SecurityDomainTag(
    "model_identity", parent=SYSTEM_TAG,
)
"""Knowledge of which LLM the agent uses.  Sibling of prompt/tool_catalogue
so that 'attacker knows the victim model' can be modeled independently
of any write capability."""

AGENT_TRACE_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace", parent=SYSTEM_TAG,
)
"""Aggregate read access to the agent's runtime trace.  Implies the three
finer-grained children."""

AGENT_TRACE_MESSAGES_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_messages", parent=AGENT_TRACE_TAG,
)
"""Read access to the agent's chat-message stream."""

AGENT_TRACE_TOOL_CALLS_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_tool_calls", parent=AGENT_TRACE_TAG,
)
"""Read access to the function calls the agent emits."""

AGENT_TRACE_TOOL_RESPONSES_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_tool_responses", parent=AGENT_TRACE_TAG,
)
"""Read access to the tool return values the agent observes (after any
on-demand injection has been applied)."""

# ---------------------------------------------------------------------------
# Tree 2: user
# ---------------------------------------------------------------------------

USER_TAG: SecurityDomainTag = SecurityDomainTag("user")
"""User prompt channel.  An attacker with ``user`` can override the
benign user prompt set by the Task via the user-prompt Controllable."""

# ---------------------------------------------------------------------------
# Tree 3: tools (2x2 grid)
# ---------------------------------------------------------------------------

TOOLS_TAG: SecurityDomainTag = SecurityDomainTag("tools")
"""Root of the tool-content surface.  An attacker with ``tools`` can
inject into every readable tool's return value regardless of provenance."""

CONTENT_1P_DATA_1P_TAG: SecurityDomainTag = SecurityDomainTag(
    "content_1p_data_1p", parent=TOOLS_TAG,
)
"""Content authored by the first party (user / our system), stored in
first-party storage (e.g. the user's own filesystem).  Examples:
``banking.get_balance``, ``workspace.get_current_day``."""

CONTENT_1P_DATA_3P_TAG: SecurityDomainTag = SecurityDomainTag(
    "content_1p_data_3p", parent=TOOLS_TAG,
)
"""First-party content held in third-party storage (e.g. user's PII held
by the travel agency, the user's sent emails held by the mail provider).
Examples: ``travel.get_user_information``, ``workspace.get_sent_emails``."""

CONTENT_3P_DATA_1P_TAG: SecurityDomainTag = SecurityDomainTag(
    "content_3p_data_1p", parent=TOOLS_TAG,
)
"""Third-party content held in first-party storage (e.g. a third-party
vendor's bill stored in the user's filesystem).  Examples:
``banking.read_file('bill-december-2023.txt')``."""

CONTENT_3P_DATA_3P_TAG: SecurityDomainTag = SecurityDomainTag(
    "content_3p_data_3p", parent=TOOLS_TAG,
)
"""Third-party content in third-party storage (e.g. hotel reviews on a
booking aggregator, emails received from external senders).  Examples:
``travel.get_rating_reviews_for_hotels``, ``slack.get_webpage``."""

# ---------------------------------------------------------------------------
# Assembled SecurityDomain
# ---------------------------------------------------------------------------

DOMAIN: SecurityDomain = SecurityDomain([
    # system tree
    SYSTEM_TAG,
    PROMPT_TAG,
    TOOL_CATALOGUE_TAG, TOOL_CATALOGUE_ADDABLE_TAG,
    MODEL_IDENTITY_TAG,
    AGENT_TRACE_TAG,
    AGENT_TRACE_MESSAGES_TAG, AGENT_TRACE_TOOL_CALLS_TAG, AGENT_TRACE_TOOL_RESPONSES_TAG,
    # user tree
    USER_TAG,
    # tools tree
    TOOLS_TAG,
    CONTENT_1P_DATA_1P_TAG, CONTENT_1P_DATA_3P_TAG,
    CONTENT_3P_DATA_1P_TAG, CONTENT_3P_DATA_3P_TAG,
])
"""The full security-domain forest exposed by :class:`AgentDojoTarget`.
15 tags across three trees."""


__all__ = [
    # system tree
    "SYSTEM_TAG",
    "PROMPT_TAG",
    "TOOL_CATALOGUE_TAG", "TOOL_CATALOGUE_ADDABLE_TAG",
    "MODEL_IDENTITY_TAG",
    "AGENT_TRACE_TAG",
    "AGENT_TRACE_MESSAGES_TAG", "AGENT_TRACE_TOOL_CALLS_TAG", "AGENT_TRACE_TOOL_RESPONSES_TAG",
    # user tree
    "USER_TAG",
    # tools tree
    "TOOLS_TAG",
    "CONTENT_1P_DATA_1P_TAG", "CONTENT_1P_DATA_3P_TAG",
    "CONTENT_3P_DATA_1P_TAG", "CONTENT_3P_DATA_3P_TAG",
    # assembled forest
    "DOMAIN",
]
