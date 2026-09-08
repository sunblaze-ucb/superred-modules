"""Past/future tense reformulation optimizer for superred.

Based on Tencent Zhuque Lab AI-Infra-Guard
(https://github.com/Tencent/AI-Infra-Guard).
"""

from past_tense_optimizer.optimizer import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TENSE,
    PastTenseOptimizer,
)

__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TENSE",
    "PastTenseOptimizer",
]
