"""System-prompt-extraction optimizer for superred, ported from garak."""

from sysprompt_extraction_optimizer.optimizer import (
    ATTACK_CATEGORIES,
    SysPromptExtractionOptimizer,
    load_attack_templates,
)

__all__ = [
    "ATTACK_CATEGORIES",
    "SysPromptExtractionOptimizer",
    "load_attack_templates",
]
