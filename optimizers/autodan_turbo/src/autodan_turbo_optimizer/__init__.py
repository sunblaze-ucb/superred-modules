"""AutoDAN-Turbo lifelong-strategy jailbreak optimizer for superred.

Reference: Liu et al., "AutoDAN-Turbo: A Lifelong Agent for Strategy
Self-Exploration to Jailbreak LLMs," arXiv:2410.05295 (ICLR 2025
spotlight); the official ``SaFoLab-WISC/AutoDAN-Turbo`` reference
implementation.
"""

from autodan_turbo_optimizer.optimizer import AutoDANTurboOptimizer

__all__ = ["AutoDANTurboOptimizer"]
