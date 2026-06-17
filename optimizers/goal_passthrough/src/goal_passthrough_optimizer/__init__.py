"""Goal-passthrough baseline optimizer for superred.

The unattacked direct-prompt baseline: injects the raw ``Goal`` text once,
then stops. Used as the zero-attack reference point in attacker sweeps.
"""

from goal_passthrough_optimizer.optimizer import GoalPassthroughOptimizer

__all__ = ["GoalPassthroughOptimizer"]
