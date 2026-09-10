"""False-assertion optimizer for superred, ported from garak."""

from false_assertion_optimizer.optimizer import (
    PREFIXES,
    FalseAssertionOptimizer,
    build_prompts,
    load_false_claims,
)

__all__ = [
    "FalseAssertionOptimizer",
    "PREFIXES",
    "build_prompts",
    "load_false_claims",
]
