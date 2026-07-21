"""AutoDAN-Turbo lifelong-strategy jailbreak optimizer for superred.

Reference: Liu et al., "AutoDAN-Turbo: A Lifelong Agent for Strategy
Self-Exploration to Jailbreak LLMs," arXiv:2410.05295 (ICLR 2025
spotlight); the official ``SaFo-Lab/AutoDAN-Turbo`` reference
implementation (GitHub org renamed from ``SaFoLab-WISC`` since the paper's
release).
"""

from autodan_turbo_optimizer.optimizer import AutoDANTurboOptimizer

__all__ = ["AutoDANTurboOptimizer"]
