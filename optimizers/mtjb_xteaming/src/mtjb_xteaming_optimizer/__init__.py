"""X-Teaming multi-turn optimizer for superred.

Ported from MT-JailBench (``engine/attacks/xteaming``). See ``README.md`` and
``ASSUMPTIONS.md``. The TextGrad refine step is optional (extra ``textgrad``).
"""

from mtjb_xteaming_optimizer.optimizer import (
    DEFAULT_MAX_JUDGE_RETRIES,
    DEFAULT_MAX_REFINES_PER_TURN,
    DEFAULT_MAX_TOTAL_TURNS,
    MtjbXTeamingOptimizer,
)

__all__ = [
    "DEFAULT_MAX_JUDGE_RETRIES",
    "DEFAULT_MAX_REFINES_PER_TURN",
    "DEFAULT_MAX_TOTAL_TURNS",
    "MtjbXTeamingOptimizer",
]
