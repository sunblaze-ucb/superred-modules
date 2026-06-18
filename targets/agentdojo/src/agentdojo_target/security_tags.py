"""SecurityDomain forest and tag instances for the AgentDojo composite target.

The forest has three independent root trees:

- ``system``: agent-side capabilities the attacker may control.  Subtree
  layout encodes capability subsumption: an attacker holding
  ``tool_catalogue`` automatically holds its three catalogue write
  capabilities ``tool_catalogue_add`` / ``tool_catalogue_edit`` /
  ``tool_catalogue_remove``.  Read-only access to a surface is not a
  separate tag; it is granted per threat model by listing the tag in the
  Controller's ``read_only`` set rather than its read & write ``scope``.
- ``user``: the user prompt channel.  A single tag (no children) because
  the user channel has no internal hierarchy in this threat model.
- ``tools``: a per-service / per-store forest mirroring the real
  systems the agent drives.  ``tools`` is a pure grouping root (nothing
  is emitted directly at it); under it sit the four service nodes
  (``banking``, ``workspace``, ``slack``, ``travel``) and, under each,
  one leaf per separately-compromisable data store of that service
  (e.g. ``banking_bank_account``, ``workspace_inbox``).  Each leaf maps
  to exactly one upstream pydantic store.  Reading from a store and the
  agent action that mutates that same store share the store's leaf, so
  granting one service label covers both the reads an attacker can
  inject and the actions it provokes against that service.  Granting a
  service grants all its stores; granting ``tools`` grants everything.

Tag names are service-qualified for global uniqueness (the workspace
and travel suites each own a distinct inbox/calendar instance).  Each
tag is a module-level
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

SYSTEM_PROMPT_TAG: SecurityDomainTag = SecurityDomainTag("system_prompt", parent=SYSTEM_TAG)
"""System prompt surface.  When ``system_prompt`` is in the Controller's read &
write ``scope`` the attacker may override it; listing it under
``read_only`` instead grants see-but-not-change access."""

TOOL_CATALOGUE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue",
    parent=SYSTEM_TAG,
)
"""Grouping root of the tool-catalogue (registry) capability.  Holding it
subsumes all three catalogue write capabilities below — add, edit, and
remove.  The catalogue-listing observable carries this tag, so listing
``tool_catalogue`` under ``read_only`` (rather than ``scope``) grants the
listing without any edit capability."""

TOOL_CATALOGUE_ADD_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_add",
    parent=TOOL_CATALOGUE_TAG,
)
"""Register-only catalogue capability: the attacker may ADD new
(attacker-defined) tools but cannot edit or remove existing entries.
Models a malicious-MCP that can only contribute additional tools, not
shadow or delete existing ones."""

TOOL_CATALOGUE_EDIT_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_edit",
    parent=TOOL_CATALOGUE_TAG,
)
"""Edit-existing catalogue capability: the attacker may EDIT an existing
tool — both its implementation/callback (replace, which shadows the tool's
behaviour with an attacker-controlled return) and its description
(rewrite-description) — but cannot add or remove entries."""

TOOL_CATALOGUE_REMOVE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_remove",
    parent=TOOL_CATALOGUE_TAG,
)
"""Remove catalogue capability: the attacker may UNREGISTER (remove) an
existing tool, but cannot add or edit entries."""

MODEL_IDENTITY_TAG: SecurityDomainTag = SecurityDomainTag(
    "model_identity",
    parent=SYSTEM_TAG,
)
"""Knowledge of which LLM the agent uses.  Sibling of prompt/tool_catalogue
so that 'attacker knows the victim model' can be modeled independently
of any write capability."""

AGENT_TRACE_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace",
    parent=SYSTEM_TAG,
)
"""Aggregate read access to the agent's runtime trace.  Carries the
non-tool internal message stream only (see :data:`AGENT_TRACE_MESSAGES_TAG`).
Tool calls and tool responses are NOT mirrored here: each tool's call
(function + arguments) and return value are emitted exactly once, on that
tool's :class:`ControllablePostCallEvent` (read via the tool's own ``tools``
leaf), so there is no duplicate agent-trace surface for them."""

AGENT_TRACE_MESSAGES_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_messages",
    parent=AGENT_TRACE_TAG,
)
"""Read access to the agent's NON-TOOL chat-message stream (system,
assistant reasoning, and user messages).  Tool-call requests and
tool-result messages are excluded — that information lives once on the
per-tool ``ControllablePostCallEvent``."""

# ---------------------------------------------------------------------------
# Tree 2: user
# ---------------------------------------------------------------------------

USER_TAG: SecurityDomainTag = SecurityDomainTag("user")
"""User prompt channel.  An attacker with ``user`` can override the
benign user prompt set by the Task via the user-prompt Controllable."""

# ---------------------------------------------------------------------------
# Tree 3: tools (per-service / per-store forest)
#
# ``tools`` is a pure grouping root: nothing is ever emitted at it.  Under
# it sit four service nodes, and under each service one leaf per
# separately-compromisable upstream data store (one leaf == one pydantic
# store == one ``Depends`` extractor).  A read controllable and the write
# observation for the same store reference the same leaf, so reading and
# acting on a store share a boundary.
# ---------------------------------------------------------------------------

TOOLS_TAG: SecurityDomainTag = SecurityDomainTag("tools")
"""Grouping root of the tool surface.  An attacker granted ``tools`` can
inject into every readable store and observe every action across all four
services.  Nothing is emitted directly at this node; every surface sits at
a service node or a store leaf below it."""

# ---- banking (BankingEnvironment: bank_account, filesystem, user_account) --

BANKING_TAG: SecurityDomainTag = SecurityDomainTag("banking", parent=TOOLS_TAG)
"""The banking service.  Granting it covers all banking stores below."""

BANKING_BANK_ACCOUNT_TAG: SecurityDomainTag = SecurityDomainTag(
    "banking_bank_account", parent=BANKING_TAG
)
"""Balance, IBAN, transaction history, and scheduled transactions."""

BANKING_FILESYSTEM_TAG: SecurityDomainTag = SecurityDomainTag(
    "banking_filesystem", parent=BANKING_TAG
)
"""Files readable on the banking side (bills, letters)."""

BANKING_USER_ACCOUNT_TAG: SecurityDomainTag = SecurityDomainTag(
    "banking_user_account", parent=BANKING_TAG
)
"""The account holder's profile and password."""

# ---- workspace (WorkspaceEnvironment: inbox, calendar, cloud_drive) --------

WORKSPACE_TAG: SecurityDomainTag = SecurityDomainTag("workspace", parent=TOOLS_TAG)
"""The office-suite service (email + calendar + files)."""

WORKSPACE_INBOX_TAG: SecurityDomainTag = SecurityDomainTag("workspace_inbox", parent=WORKSPACE_TAG)
"""The mailbox: emails and contacts."""

WORKSPACE_CALENDAR_TAG: SecurityDomainTag = SecurityDomainTag(
    "workspace_calendar", parent=WORKSPACE_TAG
)
"""The calendar: events."""

WORKSPACE_CLOUD_DRIVE_TAG: SecurityDomainTag = SecurityDomainTag(
    "workspace_cloud_drive", parent=WORKSPACE_TAG
)
"""The file storage."""

# ---- slack (SlackEnvironment: slack, web) ----------------------------------

SLACK_TAG: SecurityDomainTag = SecurityDomainTag("slack", parent=TOOLS_TAG)
"""The team-chat service."""

SLACK_SLACK_TAG: SecurityDomainTag = SecurityDomainTag("slack_slack", parent=SLACK_TAG)
"""The messaging store: users, channels, channel messages, and direct
messages.  Upstream holds these in one ``Slack`` object reached by a
single ``Depends('slack')``, so channel-messages and direct-messages are
not separable into finer leaves (see ``ASSUMPTIONS.md``)."""

SLACK_WEB_TAG: SecurityDomainTag = SecurityDomainTag("slack_web", parent=SLACK_TAG)
"""Fetched web pages and the web-request log."""

# ---- travel (TravelEnvironment: hotels, restaurants, car_rental, flights,
#      user, calendar, reservation, inbox) -----------------------------------

TRAVEL_TAG: SecurityDomainTag = SecurityDomainTag("travel", parent=TOOLS_TAG)
"""The travel-booking service."""

TRAVEL_HOTELS_TAG: SecurityDomainTag = SecurityDomainTag("travel_hotels", parent=TRAVEL_TAG)
"""Hotel listings."""

TRAVEL_RESTAURANTS_TAG: SecurityDomainTag = SecurityDomainTag(
    "travel_restaurants", parent=TRAVEL_TAG
)
"""Restaurant listings."""

TRAVEL_CAR_RENTAL_TAG: SecurityDomainTag = SecurityDomainTag("travel_car_rental", parent=TRAVEL_TAG)
"""Car-rental listings."""

TRAVEL_FLIGHTS_TAG: SecurityDomainTag = SecurityDomainTag("travel_flights", parent=TRAVEL_TAG)
"""Flight listings."""

TRAVEL_USER_TAG: SecurityDomainTag = SecurityDomainTag("travel_user", parent=TRAVEL_TAG)
"""The traveller's saved profile (PII)."""

TRAVEL_CALENDAR_TAG: SecurityDomainTag = SecurityDomainTag("travel_calendar", parent=TRAVEL_TAG)
"""The travel-side calendar (a distinct instance from the workspace one)."""

TRAVEL_RESERVATION_TAG: SecurityDomainTag = SecurityDomainTag(
    "travel_reservation", parent=TRAVEL_TAG
)
"""The pending reservation (hotel, restaurant, or car)."""

TRAVEL_INBOX_TAG: SecurityDomainTag = SecurityDomainTag("travel_inbox", parent=TRAVEL_TAG)
"""The travel-side mailbox (a distinct instance from the workspace one)."""

# ---------------------------------------------------------------------------
# Assembled SecurityDomain
# ---------------------------------------------------------------------------

DOMAIN: SecurityDomain = SecurityDomain(
    [
        # system tree
        SYSTEM_TAG,
        SYSTEM_PROMPT_TAG,
        TOOL_CATALOGUE_TAG,
        TOOL_CATALOGUE_ADD_TAG,
        TOOL_CATALOGUE_EDIT_TAG,
        TOOL_CATALOGUE_REMOVE_TAG,
        MODEL_IDENTITY_TAG,
        AGENT_TRACE_TAG,
        AGENT_TRACE_MESSAGES_TAG,
        # user tree
        USER_TAG,
        # tools tree: grouping root + 4 services + 16 store leaves
        TOOLS_TAG,
        BANKING_TAG,
        BANKING_BANK_ACCOUNT_TAG,
        BANKING_FILESYSTEM_TAG,
        BANKING_USER_ACCOUNT_TAG,
        WORKSPACE_TAG,
        WORKSPACE_INBOX_TAG,
        WORKSPACE_CALENDAR_TAG,
        WORKSPACE_CLOUD_DRIVE_TAG,
        SLACK_TAG,
        SLACK_SLACK_TAG,
        SLACK_WEB_TAG,
        TRAVEL_TAG,
        TRAVEL_HOTELS_TAG,
        TRAVEL_RESTAURANTS_TAG,
        TRAVEL_CAR_RENTAL_TAG,
        TRAVEL_FLIGHTS_TAG,
        TRAVEL_USER_TAG,
        TRAVEL_CALENDAR_TAG,
        TRAVEL_RESERVATION_TAG,
        TRAVEL_INBOX_TAG,
    ]
)
"""The full security-domain forest exposed by :class:`AgentDojoTarget`.
31 tags across three trees (system 9, user 1, tools 21)."""


__all__ = [
    # system tree
    "SYSTEM_TAG",
    "SYSTEM_PROMPT_TAG",
    "TOOL_CATALOGUE_TAG",
    "TOOL_CATALOGUE_ADD_TAG",
    "TOOL_CATALOGUE_EDIT_TAG",
    "TOOL_CATALOGUE_REMOVE_TAG",
    "MODEL_IDENTITY_TAG",
    "AGENT_TRACE_TAG",
    "AGENT_TRACE_MESSAGES_TAG",
    # user tree
    "USER_TAG",
    # tools tree: grouping root + services + store leaves
    "TOOLS_TAG",
    "BANKING_TAG",
    "BANKING_BANK_ACCOUNT_TAG",
    "BANKING_FILESYSTEM_TAG",
    "BANKING_USER_ACCOUNT_TAG",
    "WORKSPACE_TAG",
    "WORKSPACE_INBOX_TAG",
    "WORKSPACE_CALENDAR_TAG",
    "WORKSPACE_CLOUD_DRIVE_TAG",
    "SLACK_TAG",
    "SLACK_SLACK_TAG",
    "SLACK_WEB_TAG",
    "TRAVEL_TAG",
    "TRAVEL_HOTELS_TAG",
    "TRAVEL_RESTAURANTS_TAG",
    "TRAVEL_CAR_RENTAL_TAG",
    "TRAVEL_FLIGHTS_TAG",
    "TRAVEL_USER_TAG",
    "TRAVEL_CALENDAR_TAG",
    "TRAVEL_RESERVATION_TAG",
    "TRAVEL_INBOX_TAG",
    # assembled forest
    "DOMAIN",
]
