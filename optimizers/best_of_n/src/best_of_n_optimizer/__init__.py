"""Best-of-N jailbreaking optimizer for superred (text arm, ported from bon-jailbreaking)."""

from best_of_n_optimizer.augmentations import (
    DEFAULT_SIGMA,
    apply_ascii_noising,
    apply_random_capitalization,
    apply_word_scrambling,
    augment,
)
from best_of_n_optimizer.optimizer import DEFAULT_N_SAMPLES, BestOfNOptimizer

__all__ = [
    "DEFAULT_N_SAMPLES",
    "DEFAULT_SIGMA",
    "BestOfNOptimizer",
    "apply_ascii_noising",
    "apply_random_capitalization",
    "apply_word_scrambling",
    "augment",
]
