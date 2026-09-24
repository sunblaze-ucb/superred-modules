"""Chain-of-Attack (CoA) multi-turn optimizer for superred.

Ported from MT-JailBench (``engine/attacks/coa``), itself adapted from
YancyKahn/CoA. See ``README.md`` and ``ASSUMPTIONS.md``.
"""

from mtjb_coa_optimizer.optimizer import (
    DEFAULT_MAX_JUDGE_RETRIES,
    DEFAULT_MAX_TURNS,
    DEFAULT_MAX_UPDATE_RETRIES,
    DEFAULT_N_INIT_CHAINS,
    DEFAULT_SEMANTIC_UPDATE_SLACK,
    MtjbCoAOptimizer,
)

__all__ = [
    "DEFAULT_MAX_JUDGE_RETRIES",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MAX_UPDATE_RETRIES",
    "DEFAULT_N_INIT_CHAINS",
    "DEFAULT_SEMANTIC_UPDATE_SLACK",
    "MtjbCoAOptimizer",
]
