"""API-key-leak optimizer for superred, ported from garak."""

from apikey_leak_optimizer.optimizer import (
    COMPLETE_TEMPLATE,
    GENERATE_TEMPLATE,
    KEY_TYPES,
    PARTIAL_KEYS,
    ApiKeyLeakOptimizer,
    build_prompts,
)

__all__ = [
    "ApiKeyLeakOptimizer",
    "COMPLETE_TEMPLATE",
    "GENERATE_TEMPLATE",
    "KEY_TYPES",
    "PARTIAL_KEYS",
    "build_prompts",
]
