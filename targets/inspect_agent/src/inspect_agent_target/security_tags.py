"""SecurityDomain forest for the general inspect-agent target.

Structured in the AgentDojo style: independent root trees, with capability
subsumption encoded by the parent/child hierarchy (a scope holding a parent tag
includes all its descendants, so the Controller can scope broadly or narrowly).

- ``system``: agent-side surfaces (everything the agent IS / how it is
  configured / its own trace).  An attacker with ``system`` holds every
  agent-side capability below it.
    - ``system_prompt``: the prompt surface.  Read-only access is not a
      separate tag; it is granted per threat model by listing the tag in the
      Controller's ``read_only`` set rather than its read & write ``scope``.
    - ``tool_catalogue`` (the *registry*: which tools exist and their docs) is a
      grouping root over three sibling write capabilities:
      ``tool_catalogue_add`` (register a new tool), ``tool_catalogue_edit``
      (replace an implementation or rewrite a description), and
      ``tool_catalogue_remove`` (unregister a tool).  The catalogue-listing
      observable carries ``tool_catalogue`` itself, so listing it under
      ``read_only`` (rather than ``scope``) grants the listing without edit
      capability.
    - ``model_identity``: which model powers the agent.
    - ``agent_trace`` (read the run trace) -> ``agent_trace_messages``: the
      NON-TOOL internal message stream (system / assistant reasoning / user).
      Tool calls and tool responses are NOT mirrored here; each is emitted once
      on the tool's ``ControllablePostCallEvent`` (read via the tool's ``tools``
      leaf).
- ``user``: the user-prompt channel (the jailbreak / prompt-attack surface).
- ``tools``: one write surface per tool the agent can call.  Injecting here
  replaces what that tool returns to the agent (indirect prompt injection).
  The ``tools`` root carries NO children by itself; a SecurityClaim supplies a
  per-tool *trust-boundary* sub-forest (e.g. web / social / financial) parented
  under ``tools``, plus a tool->tag map, via the target constructor.  With no
  claim-supplied scopes every tool falls back to the bare ``tools`` root.  A
  tool's call (function + arguments) and its return are emitted once, on that
  tool's ``ControllablePostCallEvent``, readable via the tool's own ``tools``
  leaf under ``read_only``.

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
    "system_prompt",
    parent=SYSTEM_TAG,
)
"""Agent system prompt surface.  When ``system_prompt`` is in the Controller's
read & write ``scope`` the attacker may override it; listing it under
``read_only`` instead grants see-but-not-change access."""

# --- tool catalogue (the tool registry) ------------------------------------

TOOL_CATALOGUE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue",
    parent=SYSTEM_TAG,
)
"""Grouping root of the tool-registry capability: holding it subsumes all three
catalogue write capabilities below — add, edit, and remove.  The
catalogue-listing observable carries this tag, so listing ``tool_catalogue``
under ``read_only`` (rather than ``scope``) grants the listing without any edit
capability.  This is the *registry* (which tools exist), distinct from
:data:`TOOLS_TAG` (what a tool returns)."""

TOOL_CATALOGUE_ADD_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_add",
    parent=TOOL_CATALOGUE_TAG,
)
"""Register-only catalogue capability: the attacker may ADD new
(attacker-defined) tools but cannot edit or remove existing ones.  Models a
malicious-MCP that only contributes extra tools."""

TOOL_CATALOGUE_EDIT_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_edit",
    parent=TOOL_CATALOGUE_TAG,
)
"""Edit-existing catalogue capability: the attacker may EDIT an existing tool —
both its implementation/callback (replace, which shadows the tool's behaviour
with an attacker-controlled return) and its description (rewrite-description) —
but cannot add or remove entries."""

TOOL_CATALOGUE_REMOVE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_remove",
    parent=TOOL_CATALOGUE_TAG,
)
"""Remove catalogue capability: the attacker may UNREGISTER (remove) an existing
tool, but cannot add or edit entries."""

# --- model identity --------------------------------------------------------

MODEL_IDENTITY_TAG: SecurityDomainTag = SecurityDomainTag(
    "model_identity",
    parent=SYSTEM_TAG,
)
"""Knowledge of which model the agent uses."""

MESSAGE_LIMIT_TAG: SecurityDomainTag = SecurityDomainTag(
    "message_limit",
    parent=SYSTEM_TAG,
)
"""The rollout's message-limit surface (the message_limit static observable).
There is no message-limit controllable, so the tag grants observation only,
at any access level.  The message_limit config slot itself is scoped to the
broader ``system`` root."""

# --- agent trace (read access) ---------------------------------------------

AGENT_TRACE_TAG: SecurityDomainTag = SecurityDomainTag("agent_trace", parent=SYSTEM_TAG)
"""Aggregate read access to the agent's run trace (the non-tool internal message
stream).  Tool calls and tool responses are NOT mirrored here: each tool's call
(function + arguments) and return value are emitted exactly once, on that tool's
:class:`ControllablePostCallEvent` (read via the tool's own ``tools`` leaf), so
there is no duplicate agent-trace surface for them."""

AGENT_TRACE_MESSAGES_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_messages",
    parent=AGENT_TRACE_TAG,
)
"""Read access to the agent's NON-TOOL chat-message stream (system, assistant
reasoning, and user messages).  Tool-call requests and tool-result messages are
excluded -- that information lives once on the per-tool
``ControllablePostCallEvent``."""

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
sub-forest every tool falls back to this bare root.  Read side: the tool's own
``ControllablePostCallEvent`` (carrying the call + return), readable via this
leaf under ``read_only``."""

# ===========================================================================
# Assembled forest
# ===========================================================================

FIXED_TAGS: tuple[SecurityDomainTag, ...] = (
    # system tree
    SYSTEM_TAG,
    SYSTEM_PROMPT_TAG,
    TOOL_CATALOGUE_TAG,
    TOOL_CATALOGUE_ADD_TAG,
    TOOL_CATALOGUE_EDIT_TAG,
    TOOL_CATALOGUE_REMOVE_TAG,
    MODEL_IDENTITY_TAG,
    MESSAGE_LIMIT_TAG,
    AGENT_TRACE_TAG,
    AGENT_TRACE_MESSAGES_TAG,
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
    "TOOL_CATALOGUE_TAG",
    "TOOL_CATALOGUE_ADD_TAG",
    "TOOL_CATALOGUE_EDIT_TAG",
    "TOOL_CATALOGUE_REMOVE_TAG",
    "MODEL_IDENTITY_TAG",
    "MESSAGE_LIMIT_TAG",
    "AGENT_TRACE_TAG",
    "AGENT_TRACE_MESSAGES_TAG",
    "USER_TAG",
    "TOOLS_TAG",
    "FIXED_TAGS",
    "tool_tag_closure",
    "build_domain",
    "DOMAIN",
]
