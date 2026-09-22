"""ReNeLLM generalized nested-jailbreak optimizer for superred.

Ported from ``NJUNLP/ReNeLLM`` (Ding et al., "A Wolf in Sheep's Clothing:
Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily",
NAACL 2024, arXiv:2311.08268). See ``README.md`` and ``ASSUMPTIONS.md``.
"""

from renellm_optimizer.optimizer import (
    DEFAULT_ITER_MAX,
    DEFAULT_MAX_REWRITE_ATTEMPTS,
    ReNeLLMOptimizer,
)

__all__ = [
    "DEFAULT_ITER_MAX",
    "DEFAULT_MAX_REWRITE_ATTEMPTS",
    "ReNeLLMOptimizer",
]
