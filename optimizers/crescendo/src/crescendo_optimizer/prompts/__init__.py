"""Crescendo attacker prompt variants."""

from crescendo_optimizer.prompts.variant_1 import SYSTEM_PROMPT as _V1
from crescendo_optimizer.prompts.variant_2 import SYSTEM_PROMPT as _V2
from crescendo_optimizer.prompts.variant_3 import SYSTEM_PROMPT as _V3

_VARIANTS = [_V1, _V2, _V3]


def get_variant(index: int) -> str:
    """Return the prompt variant at the given index, cycling if out of range."""
    return _VARIANTS[index % len(_VARIANTS)]


def get_variant_count() -> int:
    """Return the number of available prompt variants."""
    return len(_VARIANTS)
