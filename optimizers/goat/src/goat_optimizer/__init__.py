"""GOAT (Generative Offensive Agent Tester) multi-turn jailbreak optimizer.

Reference: Pavlova et al., "Automated Red Teaming with GOAT: the Generative
Offensive Agent Tester," arXiv:2410.01606 (Meta, 2024 / ICML 2025).
"""

from goat_optimizer.attacks import ATTACKS, Attack
from goat_optimizer.optimizer import GOATOptimizer

__all__ = ["GOATOptimizer", "Attack", "ATTACKS"]
