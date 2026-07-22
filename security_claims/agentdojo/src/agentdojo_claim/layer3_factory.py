"""Layer 3: combined claim composing Layers 1 and 2.

Trivial composition via :meth:`SecurityClaim.from_claims`.  Per-task
scores carry distinct sub-score keys (Layer 1: ``category_<label>`` +
``suite_<name>``; Layer 2: ``category_<label>``) so downstream analysis
can split by layer using the primary_score's ``name`` field
(``attack_succeeded`` for Layer 1, ``goal_achieved`` for Layer 2).
"""

from __future__ import annotations

from agentdojo_target.target import AgentDojoTarget
from superred.core.interfaces.security_claim import SecurityClaim

from agentdojo_claim.layer1_factory import agentdojo_layer1_claim
from agentdojo_claim.layer2_factory import agentdojo_layer2_claim


def agentdojo_combined_claim(
    *,
    layer1_kwargs: dict | None = None,
    layer2_kwargs: dict | None = None,
) -> SecurityClaim[AgentDojoTarget]:
    """Build the combined Layer-1 + Layer-2 claim.

    Args:
        layer1_kwargs: Forwarded to :func:`agentdojo_layer1_claim`.
        layer2_kwargs: Forwarded to :func:`agentdojo_layer2_claim`.

    Returns:
        A :class:`SecurityClaim` that iterates Layer-1 tasks first,
        then Layer-2 tasks.
    """
    layer1 = agentdojo_layer1_claim(**(layer1_kwargs or {}))
    layer2 = agentdojo_layer2_claim(**(layer2_kwargs or {}))
    return SecurityClaim.from_claims([layer1, layer2])


__all__ = ["agentdojo_combined_claim"]
