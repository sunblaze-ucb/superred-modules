"""security_claim_agentdojo: AgentDojo SecurityClaim for superred.

Three composable layers:

- Layer 1: port of AgentDojo's 27 original injection tasks.
- Layer 2: bespoke system-purpose-violation goals against the composite
  AgentDojoTarget.
- Layer 3: composition of layers 1 and 2.

Public exports are added as each module lands.  v0.1.0 is alpha.
"""

from __future__ import annotations

from security_claim_agentdojo.layer1_categories import (
    ALL_CATEGORIES,
    CATEGORIES_BY_SUITE,
    INJECTION_CATEGORIES,
    category_of,
)
from security_claim_agentdojo.layer1_factory import (
    agentdojo_layer1_banking_claim,
    agentdojo_layer1_category_claim,
    agentdojo_layer1_claim,
    agentdojo_layer1_slack_claim,
    agentdojo_layer1_suite_claim,
    agentdojo_layer1_travel_claim,
    agentdojo_layer1_workspace_claim,
)
from security_claim_agentdojo.layer1_pairs import CANONICAL_PAIRS
from security_claim_agentdojo.layer1_task import AgentDojoPairedTask
from security_claim_agentdojo.layer2_factory import (
    agentdojo_layer2_category_claim,
    agentdojo_layer2_claim,
    layer2_categories,
    layer2_goal_ids,
)
from security_claim_agentdojo.layer2_task import (
    Layer2GoalSpec,
    SecurityPredicate,
    SystemViolatingTask,
)
from security_claim_agentdojo.layer3_factory import agentdojo_combined_claim

__version__ = "0.1.0"
__all__ = [
    # Layer 1
    "AgentDojoPairedTask",
    "agentdojo_layer1_claim",
    "agentdojo_layer1_suite_claim",
    "agentdojo_layer1_category_claim",
    "agentdojo_layer1_banking_claim",
    "agentdojo_layer1_workspace_claim",
    "agentdojo_layer1_slack_claim",
    "agentdojo_layer1_travel_claim",
    "CANONICAL_PAIRS",
    "INJECTION_CATEGORIES",
    "CATEGORIES_BY_SUITE",
    "ALL_CATEGORIES",
    "category_of",
    # Layer 2
    "SystemViolatingTask",
    "Layer2GoalSpec",
    "SecurityPredicate",
    "agentdojo_layer2_claim",
    "agentdojo_layer2_category_claim",
    "layer2_categories",
    "layer2_goal_ids",
    # Layer 3
    "agentdojo_combined_claim",
]
