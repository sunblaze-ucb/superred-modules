"""Shared evaluation types for security claim evaluators.

These types originate from the contextual security paper and provide
the reusable abstractions for claim evaluation across target modules.

Moved out of ``superred.core`` because they are evaluation-layer
concerns, not target-adapter contract types.
"""

from evaluator_types.claim_types import (
    ActionRecord,
    ClaimFamily,
    ClaimVerdict,
    ContextSnapshot,
    EvidenceKind,
    ObservationRecord,
    OracleEvidence,
    VerdictStatus,
)

__all__ = [
    "ActionRecord",
    "ClaimFamily",
    "ClaimVerdict",
    "ContextSnapshot",
    "EvidenceKind",
    "ObservationRecord",
    "OracleEvidence",
    "VerdictStatus",
]
