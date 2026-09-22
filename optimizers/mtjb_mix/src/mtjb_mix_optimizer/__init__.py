"""Mix composite multi-turn optimizer for superred.

Ports MT-JailBench's Mix attack (``engine/attacks/mix``): a configurable
combination of the Crescendo / Actor / CoA / FITD / X-Teaming families driven
through the vendored engine. See ``README.md`` and ``ASSUMPTIONS.md``. The
generation/refinement path requires the optional ``refine`` extra (textgrad).
"""

from mtjb_mix_optimizer.optimizer import (
    DEFAULT_GENERATOR,
    DEFAULT_JUDGE_AND_FLOW,
    DEFAULT_MAX_REFINES_PER_TURN,
    DEFAULT_MAX_RESTARTS,
    DEFAULT_MAX_TOTAL_TURNS,
    DEFAULT_MAX_TURNS,
    DEFAULT_UPDATER,
    VALID_ROLES,
    MtjbMixOptimizer,
)

__all__ = [
    "DEFAULT_GENERATOR",
    "DEFAULT_JUDGE_AND_FLOW",
    "DEFAULT_MAX_REFINES_PER_TURN",
    "DEFAULT_MAX_RESTARTS",
    "DEFAULT_MAX_TOTAL_TURNS",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_UPDATER",
    "VALID_ROLES",
    "MtjbMixOptimizer",
]
