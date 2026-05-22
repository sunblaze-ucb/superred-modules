"""Catalogue of Layer-2 system-violating goals.

Each submodule defines one or more :class:`Layer2GoalSpec` instances at
module scope and adds them to :data:`GOAL_SPECS`.  Importing
:mod:`layer2_goals` triggers each submodule's import and aggregates the
specs.

v1 starter catalogue: 4 intra-suite goals (one per suite) that exercise
the design end-to-end.  Additional cross-suite and capability-misuse
goals can be added by writing new submodules and appending their
specs to :data:`GOAL_SPECS` here.
"""

from __future__ import annotations

from security_claim_agentdojo.layer2_goals import (
    banking_unauth_transfer,
    slack_channel_exfiltration,
    travel_pii_exfiltration,
    workspace_email_exfiltration,
)
from security_claim_agentdojo.layer2_task import Layer2GoalSpec

GOAL_SPECS: list[Layer2GoalSpec] = [
    banking_unauth_transfer.SPEC,
    workspace_email_exfiltration.SPEC,
    slack_channel_exfiltration.SPEC,
    travel_pii_exfiltration.SPEC,
]
"""All Layer-2 goal specs registered in v1, in stable order."""


__all__ = ["GOAL_SPECS"]
