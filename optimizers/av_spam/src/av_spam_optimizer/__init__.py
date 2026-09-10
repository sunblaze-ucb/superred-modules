"""AV/anti-spam signature-smuggling optimizer for superred, ported from garak."""

from av_spam_optimizer.optimizer import (
    REPEAT_PROMPTS,
    SIGNATURES,
    AvSpamOptimizer,
    build_prompts,
)

__all__ = ["REPEAT_PROMPTS", "SIGNATURES", "AvSpamOptimizer", "build_prompts"]
