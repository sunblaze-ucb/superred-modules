"""Token-smuggling optimizer for superred (ported from NVIDIA garak)."""

from token_smuggling_optimizer.optimizer import TokenSmugglingOptimizer
from token_smuggling_optimizer.smuggling import (
    DEFAULT_HOMOGLYPH_MAP,
    HYPOTHETICAL_TEMPLATE,
    homoglyph_replace,
    hypothetical_wrap,
)

__all__ = [
    "DEFAULT_HOMOGLYPH_MAP",
    "HYPOTHETICAL_TEMPLATE",
    "TokenSmugglingOptimizer",
    "homoglyph_replace",
    "hypothetical_wrap",
]
