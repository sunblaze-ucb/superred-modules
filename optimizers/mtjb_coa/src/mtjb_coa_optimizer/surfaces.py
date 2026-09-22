"""Injection-surface selection for the CoA multi-turn optimizer.

CoA delivers one self-contained user turn per round and continues a single
conversation, so it must pin exactly one controllable -- the human user's
prompt -- and inject every round into that same surface. This is the
name/value-type backstop from ``actor_attack`` (copied per package because
the optimizers are isolated wheels): never inject the reserved system prompt,
and only accept surfaces whose value type carries free-form prose. Unlike
``actor_attack`` this optimizer does not run an LLM surface classifier; it
pins the first eligible free-text surface that fires (documented in
ASSUMPTIONS.md).
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

    ``preferred`` lists the eligible surfaces seen at ``initialize``. A
    surface raised at run time that was never listed is still injected (the
    eligibility backstop) rather than silently dropping the attack, so any
    eligible free-text surface qualifies; the optimizer pins the first one
    that fires and rejects the rest for the rest of the conversation.
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
