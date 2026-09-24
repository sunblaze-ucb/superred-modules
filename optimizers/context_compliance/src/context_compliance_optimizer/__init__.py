"""Context Compliance Attack (CCA) optimizer for superred.

Ported from microsoft/PyRIT's ``context_compliance`` technique (MIT), pinned at
commit ``2016c4a``. CCA is credited to Mark Russinovich & Ahmed Salem,
"Jailbreaking is (Mostly) Simpler Than You Think" (arXiv:2503.05264). See
``README.md`` and ``ASSUMPTIONS.md``.
"""

from context_compliance_optimizer.optimizer import (
    DEFAULT_NUM_TURNS,
    UPSTREAM_FINAL_USER_MESSAGE,
    ContextComplianceOptimizer,
)

__all__ = [
    "DEFAULT_NUM_TURNS",
    "UPSTREAM_FINAL_USER_MESSAGE",
    "ContextComplianceOptimizer",
]
