"""SecurityDomain forest for the general inspect-agent target.

Structured in the AgentDojo style: independent root trees, with capability
subsumption encoded by the parent/child hierarchy (a scope holding a parent tag
includes all its descendants, so the Controller can scope broadly or narrowly).

- ``system``: agent-side surfaces (everything the agent IS / how it is
  configured / its own trace).  An attacker with ``system`` holds every
  agent-side capability below it.
    - ``system_prompt`` (writable) -> ``system_prompt_readable`` (read-only).
    - ``tool_catalogue`` (the *registry*: which tools exist and their docs;
      broad write = set tools / replace / unregister / rewrite-description) ->
      ``tool_catalogue_readable`` (read the listing) and
      ``tool_catalogue_addable`` (register-only, the weakest write).
    - ``model_identity``: which model powers the agent.
    - ``agent_trace`` (read the run trace) -> ``agent_trace_messages`` (the full
      transcript), which subsumes the two narrower projections it embeds:
      ``agent_trace_tool_calls`` and ``agent_trace_tool_responses`` (siblings;
      a call's args and a return value are disjoint, so neither subsumes the other).
- ``user``: the user-prompt channel (the jailbreak / prompt-attack surface).
- ``tools``: one write surface per tool the agent can call.  Injecting here
  replaces what that tool returns to the agent (indirect prompt injection).
  The ``tools`` root carries NO children by itself; a SecurityClaim supplies a
  per-tool *trust-boundary* sub-forest (e.g. web / social / financial) parented
  under ``tools``, plus a tool->tag map, via the target constructor.  With no
  claim-supplied scopes every tool falls back to the bare ``tools`` root.  The
  *read* side of a tool return lives under
  ``agent_trace.agent_trace_tool_responses``.

Each tag is a module-level singleton so ``scope_includes`` compares by identity.
"""

from __future__ import annotations

from collections.abc import Iterable

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
:data:`TOOL_CATALOGUE_ADDABLE_TAG`.  This is the *registry* (which tools exist),
distinct from :data:`TOOLS_TAG` (what a tool returns)."""

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

MESSAGE_LIMIT_READABLE_TAG: SecurityDomainTag = SecurityDomainTag(
    "message_limit_readable", parent=SYSTEM_TAG,
)
"""Read-only view of the rollout's message-limit (the message_limit static
observable).  The message_limit config slot itself is scoped to the broader
``system`` root."""

# --- agent trace (read access) ---------------------------------------------

AGENT_TRACE_TAG: SecurityDomainTag = SecurityDomainTag("agent_trace", parent=SYSTEM_TAG)
"""Aggregate read access to the agent's run trace.  Implies everything below."""

AGENT_TRACE_MESSAGES_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_messages", parent=AGENT_TRACE_TAG,
)
"""Read access to the agent's full chat-message transcript.  The transcript
embeds the tool calls (as fields on assistant messages) and the tool responses
(as tool messages), so this tag *subsumes* its two children -- a scope holding
``agent_trace_messages`` can already read both projections."""

AGENT_TRACE_TOOL_CALLS_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_tool_calls", parent=AGENT_TRACE_MESSAGES_TAG,
)
"""Narrow read: only the tool calls the agent emits (function + arguments).  A
projection of the transcript; sibling of (not nested under)
:data:`AGENT_TRACE_TOOL_RESPONSES_TAG` -- a call's arguments and a return value
are disjoint, so neither subsumes the other."""

AGENT_TRACE_TOOL_RESPONSES_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_tool_responses", parent=AGENT_TRACE_MESSAGES_TAG,
)
"""Narrow read: only the tool return values the agent observes (after any
tool-output injection has been applied).  A projection of the transcript;
sibling of :data:`AGENT_TRACE_TOOL_CALLS_TAG`."""

# ===========================================================================
# Tree 2: user (the user-prompt channel)
# ===========================================================================

USER_TAG: SecurityDomainTag = SecurityDomainTag("user")
"""User-prompt channel.  An attacker with ``user`` can override the benign user
prompt set by the Task (jailbreak surface)."""

# ===========================================================================
# Tree 3: tools (per-tool returned-content write surface)
# ===========================================================================

TOOLS_TAG: SecurityDomainTag = SecurityDomainTag("tools")
"""Root of the per-tool write surface: injecting under here replaces what a tool
returns to the agent (indirect prompt injection).  Distinct from
``tool_catalogue`` (the *registry*: which tools exist and their docs); ``tools``
is the *returned content*.  The root carries no children on its own -- a
SecurityClaim parents a per-tool trust-boundary sub-forest under it (see
:func:`build_domain`) and maps each tool to a leaf.  With no claim-supplied
sub-forest every tool falls back to this bare root.  Read side:
``agent_trace_tool_responses``."""

# ===========================================================================
# Assembled forest
# ===========================================================================

FIXED_TAGS: tuple[SecurityDomainTag, ...] = (
    # system tree
    SYSTEM_TAG,
    SYSTEM_PROMPT_TAG, SYSTEM_PROMPT_READABLE_TAG,
    TOOL_CATALOGUE_TAG, TOOL_CATALOGUE_READABLE_TAG, TOOL_CATALOGUE_ADDABLE_TAG,
    MODEL_IDENTITY_TAG,
    MESSAGE_LIMIT_READABLE_TAG,
    AGENT_TRACE_TAG,
    AGENT_TRACE_MESSAGES_TAG, AGENT_TRACE_TOOL_CALLS_TAG, AGENT_TRACE_TOOL_RESPONSES_TAG,
    # user tree
    USER_TAG,
    # tools root (children are claim-supplied)
    TOOLS_TAG,
)
"""Every tag the target always exposes, independent of any claim's tool scopes."""


def tool_tag_closure(tags: Iterable[SecurityDomainTag]) -> list[SecurityDomainTag]:
    """Collect *tags* plus every ancestor up to (but excluding) :data:`TOOLS_TAG`.

    A claim supplies a tool->tag map whose values are leaves of a trust-boundary
    sub-forest parented under ``tools``.  To assemble a valid
    :class:`SecurityDomain` the target needs those leaves *and* their intermediate
    parents; this returns that closure (deduped by name -- ``tools`` itself is
    already in :data:`FIXED_TAGS`).
    """
    seen: dict[str, SecurityDomainTag] = {}
    for tag in tags:
        cur: SecurityDomainTag | None = tag
        while cur is not None and cur is not TOOLS_TAG:
            seen[cur.name] = cur
            cur = cur.parent
    return list(seen.values())


def build_domain(extra_tool_tags: Iterable[SecurityDomainTag] = ()) -> SecurityDomain:
    """Assemble the full forest: the fixed trees plus a claim's tool sub-forest.

    *extra_tool_tags* are the trust-boundary tags (leaves and/or intermediate
    nodes) a SecurityClaim parents under :data:`TOOLS_TAG`.  The ancestor closure
    is taken automatically, so passing just the leaf tags (e.g. a tool->tag map's
    values) is enough.
    """
    return SecurityDomain([*FIXED_TAGS, *tool_tag_closure(extra_tool_tags)])


DOMAIN: SecurityDomain = build_domain()
"""The default forest (no claim tool scopes): the fixed trees plus the bare
``tools`` root.  A target constructed with a tool->tag map exposes a richer
domain via :func:`build_domain`."""


__all__ = [
    "SYSTEM_TAG",
    "SYSTEM_PROMPT_TAG",
    "SYSTEM_PROMPT_READABLE_TAG",
    "TOOL_CATALOGUE_TAG",
    "TOOL_CATALOGUE_READABLE_TAG",
    "TOOL_CATALOGUE_ADDABLE_TAG",
    "MODEL_IDENTITY_TAG",
    "MESSAGE_LIMIT_READABLE_TAG",
    "AGENT_TRACE_TAG",
    "AGENT_TRACE_MESSAGES_TAG",
    "AGENT_TRACE_TOOL_CALLS_TAG",
    "AGENT_TRACE_TOOL_RESPONSES_TAG",
    "USER_TAG",
    "TOOLS_TAG",
    "FIXED_TAGS",
    "tool_tag_closure",
    "build_domain",
    "DOMAIN",
]
