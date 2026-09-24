"""Anecdoctor misinformation-elicitation optimizer for superred.

Ported from PyRIT's ``AnecdoctorGenerator`` (knowledge-graph / few-shot
misinformation content generation). See ``README.md`` and ``ASSUMPTIONS.md``.
"""

from anecdoctor_optimizer.optimizer import (
    DEFAULT_CONTENT_TYPE,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_KG_RETRIES,
    AnecdoctorOptimizer,
)

__all__ = [
    "DEFAULT_CONTENT_TYPE",
    "DEFAULT_LANGUAGE",
    "DEFAULT_MAX_KG_RETRIES",
    "AnecdoctorOptimizer",
]
