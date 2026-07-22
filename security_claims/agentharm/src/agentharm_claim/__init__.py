"""agentharm_claim: the AgentHarm SecurityClaim for superred.

A faithful port of AgentHarm (harmful test_public split, 176 behaviors) that
runs against the general :class:`inspect_agent_target.InspectAgentTarget`.
Reuses upstream ``inspect_evals.agentharm`` tools, grading functions, and
judges verbatim; subdivides into 8 per-category subclaims plus the total claim.
"""

from __future__ import annotations

from agentharm_claim.categories import (
    EXPECTED_CATEGORIES,
    discover_categories,
    validate_categories,
)
from agentharm_claim.dataset_loader import (
    DATASET_REVISION,
    load_agentharm_dataset,
)
from agentharm_claim.factory import (
    DEFAULT_TARGET_MODEL,
    agentharm_category_claim,
    agentharm_claim,
    agentharm_copyright_claim,
    agentharm_cybercrime_claim,
    agentharm_disinformation_claim,
    agentharm_drugs_claim,
    agentharm_fraud_claim,
    agentharm_harassment_claim,
    agentharm_hate_claim,
    agentharm_sexual_claim,
    agentharm_target_factory,
    agentharm_total_claim,
)
from agentharm_claim.metrics import aggregate
from agentharm_claim.task import AgentHarmTask
from agentharm_claim.tool_scopes import TOOL_SCOPES

__version__ = "0.1.0"

__all__ = [
    # claim factories
    "agentharm_claim",
    "agentharm_total_claim",
    "agentharm_category_claim",
    "agentharm_copyright_claim",
    "agentharm_cybercrime_claim",
    "agentharm_disinformation_claim",
    "agentharm_drugs_claim",
    "agentharm_fraud_claim",
    "agentharm_harassment_claim",
    "agentharm_hate_claim",
    "agentharm_sexual_claim",
    # target wiring
    "agentharm_target_factory",
    "DEFAULT_TARGET_MODEL",
    "TOOL_SCOPES",
    # task + metrics
    "AgentHarmTask",
    "aggregate",
    # data + categories
    "load_agentharm_dataset",
    "DATASET_REVISION",
    "EXPECTED_CATEGORIES",
    "discover_categories",
    "validate_categories",
    "__version__",
]
