"""Multilingual (low-resource translation) jailbreak optimizer for superred.

Ported from DeepTeam (deepteam/attacks/single_turn/multilingual).
"""

from multilingual_optimizer.optimizer import (
    DEFAULT_MAX_RETRIES,
    MultilingualOptimizer,
)

__all__ = ["DEFAULT_MAX_RETRIES", "MultilingualOptimizer"]
