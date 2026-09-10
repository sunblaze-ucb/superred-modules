"""Package-hallucination optimizer for superred, ported from garak."""

from package_hallucination_optimizer.optimizer import (
    CODE_TASKS,
    LANGUAGE_NAMES,
    REAL_CODE_TASKS,
    STUB_PROMPTS,
    UNREAL_CODE_TASKS,
    PackageHallucinationOptimizer,
    build_prompts,
)

__all__ = [
    "CODE_TASKS",
    "LANGUAGE_NAMES",
    "PackageHallucinationOptimizer",
    "REAL_CODE_TASKS",
    "STUB_PROMPTS",
    "UNREAL_CODE_TASKS",
    "build_prompts",
]
