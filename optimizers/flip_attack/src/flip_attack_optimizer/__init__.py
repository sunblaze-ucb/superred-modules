"""FlipAttack single-turn jailbreak optimizer for superred."""

try:
    from flip_attack_optimizer.optimizer import FlipAttackOptimizer
except ModuleNotFoundError:
    FlipAttackOptimizer = None  # type: ignore[assignment,misc]

__all__ = ["FlipAttackOptimizer"]
