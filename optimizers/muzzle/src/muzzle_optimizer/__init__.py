"""MUZZLE optimizer: adaptive agentic indirect-prompt-injection attack (arXiv 2602.09222).

A faithful superred port of MUZZLE (gsiros/muzzle): it observes the victim agent's
trajectory, ranks injection surfaces (vessels), probes them, crafts a context-aware
payload with an offline PAIR loop, and refines from feedback. See README.md / ASSUMPTIONS.md.
"""

from __future__ import annotations

from muzzle_optimizer.optimizer import MuzzleOptimizer

__all__ = ["MuzzleOptimizer"]
