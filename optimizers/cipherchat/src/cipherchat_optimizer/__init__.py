"""CipherChat jailbreak optimizer for superred (ported from RobustNLP/CipherChat)."""

from cipherchat_optimizer.assembly import (
    CATEGORIES,
    CIPHERS,
    build_system_prompt,
    encode,
)
from cipherchat_optimizer.optimizer import CipherChatOptimizer

__all__ = [
    "CATEGORIES",
    "CIPHERS",
    "CipherChatOptimizer",
    "build_system_prompt",
    "encode",
]
