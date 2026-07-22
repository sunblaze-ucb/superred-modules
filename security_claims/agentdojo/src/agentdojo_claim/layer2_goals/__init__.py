"""Catalogue of Layer-2 system-violating goals.

Each submodule defines one :class:`Layer2GoalSpec` instance at module
scope; :data:`GOAL_SPECS` here aggregates them in a stable order.
Adding a new goal: drop a new submodule under this directory, define
its ``SPEC`` constant, and append it to :data:`GOAL_SPECS`.

Categories present in the v1 catalogue:

- **unauth_financial_action**: banking transfers.
- **credential_change**: banking password change.
- **recurring_redirect**: banking standing-order redirect.
- **data_exfiltration**: emails / channel messages / web posts to
  attacker-controlled surfaces.
- **pii_exfiltration**: a sub-flavour of exfiltration scoped to the
  travel user's PII fields.
- **data_destruction**: deletions or cancellations the user did not
  request (calendar events, emails, reservations).
- **unauthorized_share**: shares or invitations to attacker identities
  (file shares, slack invites).
- **phishing**: agent emits a phishing URL through a tool surface.
- **unauthorized_booking**: agent makes a financial booking the user
  did not request.
- **cross_suite_exfiltration**: data from one suite leaked through a
  tool surface of another suite.  Unreachable in upstream AgentDojo
  which runs one suite at a time.
- **cross_suite_logic_violation**: the agent commits a multi-suite
  action that contradicts cross-suite state (e.g. hotel city
  contradicting calendar city).
- **capability_misuse**: the agent observably invokes an attacker-
  registered/shadowed tool.
"""

from __future__ import annotations

from agentdojo_claim.layer2_goals import (
    # v1 starters (4)
    banking_unauth_transfer,
    slack_channel_exfiltration,
    travel_pii_exfiltration,
    workspace_email_exfiltration,
    # Per-suite expansions (8)
    banking_password_change,
    banking_recurring_redirect,
    workspace_bulk_email_delete,
    workspace_calendar_destruction,
    workspace_file_share_external,
    slack_external_user_invite,
    slack_phishing_via_dm,
    travel_expensive_unauth_booking,
    travel_reservation_cancellation,
    # Cross-suite (4)
    cross_banking_to_slack,
    cross_calendar_collision_booking,
    cross_pii_via_slack_web,
    cross_workspace_to_external_email,
    # Capability misuse (1)
    capability_tool_catalog_shadow,
)
from agentdojo_claim.layer2_task import Layer2GoalSpec

GOAL_SPECS: list[Layer2GoalSpec] = [
    # ---- v1 starters ----
    banking_unauth_transfer.SPEC,
    workspace_email_exfiltration.SPEC,
    slack_channel_exfiltration.SPEC,
    travel_pii_exfiltration.SPEC,
    # ---- per-suite expansions ----
    banking_password_change.SPEC,
    banking_recurring_redirect.SPEC,
    workspace_calendar_destruction.SPEC,
    workspace_bulk_email_delete.SPEC,
    workspace_file_share_external.SPEC,
    slack_external_user_invite.SPEC,
    slack_phishing_via_dm.SPEC,
    travel_expensive_unauth_booking.SPEC,
    travel_reservation_cancellation.SPEC,
    # ---- cross-suite ----
    cross_banking_to_slack.SPEC,
    cross_pii_via_slack_web.SPEC,
    cross_workspace_to_external_email.SPEC,
    cross_calendar_collision_booking.SPEC,
    # ---- capability misuse ----
    capability_tool_catalog_shadow.SPEC,
]
"""All Layer-2 goal specs registered in the catalogue, in stable order.

Counts: 4 starters + 9 per-suite expansions + 4 cross-suite + 1
capability misuse = **18 goals total**."""


__all__ = ["GOAL_SPECS"]
