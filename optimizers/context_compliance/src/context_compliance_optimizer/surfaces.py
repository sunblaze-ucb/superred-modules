"""Injection-surface selection for the context-compliance optimizer.

CCA delivers its fabricated conversation as one self-contained user turn (see
``conversation.py`` and ASSUMPTIONS.md), so the optimizer injects into a single
free-text surface that carries prose -- never the reserved system prompt. This
is the name/value-type backstop shared across the single-turn ports (copied per
package because the optimizers are isolated wheels): the fabricated prior turns
belong in the conversation the user drives, not in a Task-configured system
prompt, and a json/number surface would discard a prose transcript.

No LLM surface classifier is run; the first eligible free-text surface is used
(documented in ASSUMPTIONS.md).
"""

from __future__ import annotations

from collections.abc import Sequence

from superred.core.types.controllable import Controllable

#: Reserved name of the system prompt; never injected into, so a Task-configured
#: system prompt survives and CCA stays a conversation-context attack.
SYSTEM_PROMPT_CONTROLLABLE_NAME = "system_prompt"

#: Value types that carry free-form prose (a json/number surface would discard
#: a prose transcript).
FREE_TEXT_VALUE_TYPES = frozenset({"", "text", "str", "string", "html", "markdown"})


def is_eligible_surface(controllable: Controllable) -> bool:
    """Whether this surface could carry the fabricated-conversation user turn."""
    if controllable.name == SYSTEM_PROMPT_CONTROLLABLE_NAME:
        return False
    return controllable.value_type.lower() in FREE_TEXT_VALUE_TYPES


def preferred_surface_names(controllables: Sequence[Controllable]) -> frozenset[str]:
    """Names of every surface eligible to receive the payload."""
    return frozenset(c.name for c in controllables if is_eligible_surface(c))


def should_inject(controllable: Controllable, preferred: frozenset[str]) -> bool:
    """Whether this surface should receive the payload.

    ``preferred`` lists the eligible surfaces seen at ``initialize``. A surface
    raised at run time that was never listed is still injected (the eligibility
    backstop) rather than silently dropping the attack, so any eligible
    free-text surface qualifies.
    """
    del preferred  # kept for API symmetry; every eligible surface qualifies
    return is_eligible_surface(controllable)


__all__ = [
    "FREE_TEXT_VALUE_TYPES",
    "SYSTEM_PROMPT_CONTROLLABLE_NAME",
    "is_eligible_surface",
    "preferred_surface_names",
    "should_inject",
]
