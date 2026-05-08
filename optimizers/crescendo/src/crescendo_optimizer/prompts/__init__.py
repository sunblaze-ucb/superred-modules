"""Crescendo attacker prompt variants."""

from crescendo_optimizer.prompts.variant_1 import SYSTEM_PROMPT as _V1
from crescendo_optimizer.prompts.variant_2 import SYSTEM_PROMPT as _V2
from crescendo_optimizer.prompts.variant_3 import SYSTEM_PROMPT as _V3
from crescendo_optimizer.prompts.variant_4 import SYSTEM_PROMPT as _V4
from crescendo_optimizer.prompts.variant_5 import SYSTEM_PROMPT as _V5

_VARIANTS = [_V1, _V2, _V3, _V4, _V5]

BASE_SCHEMA: tuple[str, ...] = (
    "generated_question",
    "last_response_summary",
    "rationale_behind_jailbreak",
)
"""Canonical JSON output fields all bundled variants instruct the
attacker LLM to produce. Single source of truth for the parser
(:func:`crescendo_optimizer.attacker.Attacker._parse_response`) and
for the (c) framing override (:func:`framing_override_instruction`).
A variant adding a field must extend this tuple."""

FRAMING_FIELD = "framing_for_system_prompt"
"""Optional extra field requested by the (c) capability."""


def get_variant(index: int) -> str:
    """Return the prompt variant at the given index, cycling if out of range."""
    return _VARIANTS[index % len(_VARIANTS)]


def get_variant_count() -> int:
    """Return the number of available prompt variants."""
    return len(_VARIANTS)
