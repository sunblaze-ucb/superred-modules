"""Injection-surface selection for the ReNeLLM optimizer.

ReNeLLM delivers a single self-contained *user* prompt (a rewritten goal nested
inside a benign code/table/story scenario) and reads the model's reply. It is a
single-turn user-input attack, so it injects into a free-text user surface and
never into the reserved system prompt (which would clobber a Task-configured
system prompt and does not match the technique). This is the name/value-type
backstop shared across the optimizer ports (copied per package because each
optimizer ships as an isolated wheel).
"""

from __future__ import annotations

from collections.abc import Sequence

from superred.core.types.controllable import Controllable

#: Reserved name of the system prompt; never injected into so a Task-configured
#: system prompt survives.
SYSTEM_PROMPT_CONTROLLABLE_NAME = "system_prompt"

#: Value types that carry free-form prose (a json/number/binary surface would
#: discard the nested-scenario payload).
FREE_TEXT_VALUE_TYPES = frozenset({"", "text", "str", "string", "html", "markdown"})


def is_eligible_surface(controllable: Controllable) -> bool:
    """Whether this surface could carry a prose ReNeLLM prompt."""
    if controllable.name == SYSTEM_PROMPT_CONTROLLABLE_NAME:
        return False
    return controllable.value_type.lower() in FREE_TEXT_VALUE_TYPES


def preferred_surface_names(controllables: Sequence[Controllable]) -> frozenset[str]:
    """Names of every surface eligible to receive the payload."""
    return frozenset(c.name for c in controllables if is_eligible_surface(c))


__all__ = [
    "FREE_TEXT_VALUE_TYPES",
    "SYSTEM_PROMPT_CONTROLLABLE_NAME",
    "is_eligible_surface",
    "preferred_surface_names",
]
