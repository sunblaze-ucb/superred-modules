"""Skeleton Key single-turn jailbreak optimizer for superred.

Ported from Microsoft PyRIT (``pyrit/executor/attack/single_turn/skeleton_key.py``).
See ``README.md`` and ``ASSUMPTIONS.md``.
"""

from skeleton_key_optimizer.optimizer import (
    DEFAULT_MAX_ATTEMPTS_ON_FAILURE,
    SkeletonKeyOptimizer,
    render,
)

__all__ = [
    "DEFAULT_MAX_ATTEMPTS_ON_FAILURE",
    "SkeletonKeyOptimizer",
    "render",
]
