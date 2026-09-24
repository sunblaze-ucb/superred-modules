"""Injection-surface selection for the Skeleton Key optimizer.

Skeleton Key delivers one self-contained user turn per run: the vendored
skeleton-key preamble (and simulated acceptance) framing the objective. It must
therefore reach the model as the human user's own prompt. This is the
name/value-type backstop shared across the single-prompt optimizers (copied per
package because the optimizers are isolated wheels): never inject the reserved
system prompt, and only accept surfaces whose value type carries free-form
prose. Like ``mtjb_coa`` -- and unlike ``deepinception``/``prefill`` -- this
optimizer runs no LLM surface classifier; it injects into every eligible
free-text surface (the per-run ``_injected`` flag keeps it to the first one
that fires each run).
"""

from __future__ import annotations

from collections.abc import Sequence

from superred.core.types.controllable import Controllable

#: Reserved name of the system prompt; never injected into so a Task-configured
#: system prompt survives.
SYSTEM_PROMPT_CONTROLLABLE_NAME = "system_prompt"

#: Value types that carry free-form prose (a json/number surface would discard
#: a prose payload).
FREE_TEXT_VALUE_TYPES = frozenset({"", "text", "str", "string", "html", "markdown"})


def is_eligible_surface(controllable: Controllable) -> bool:
    """Whether this surface could carry a prose jailbreak turn."""
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
