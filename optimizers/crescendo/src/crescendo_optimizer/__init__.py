"""Crescendo multi-turn jailbreak optimizer for superred."""

try:
    from crescendo_optimizer.optimizer import CrescendoOptimizer
except ModuleNotFoundError:  # pragma: no cover – module not yet implemented
    CrescendoOptimizer = None  # type: ignore[assignment,misc]

__all__ = ["CrescendoOptimizer"]
