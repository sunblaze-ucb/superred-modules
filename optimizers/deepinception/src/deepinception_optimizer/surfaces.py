"""Injection-surface selection for single-prompt jailbreak optimizers.

Copied per-package (the optimizers are isolated wheels that cannot import one
another), alongside ``surface_llm.py``.

These attackers deliver **one self-contained prompt** that must reach the model
as the human user's own instruction -- the jailbreak scaffolding only works if
the model reads it as the request it is answering. A name table alone cannot
find that surface on an arbitrary target, so
``surface_llm.classify_controllables`` reads each surface's own description and
says which one is the user's prompt. The name/value-type rules below remain as
the backstop, used verbatim when classification is unavailable (no LLM budget,
malformed reply, any failure -> ``{}``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from superred.core.types.controllable import Controllable

#: Categories offered to the classifier. ``user-prompt`` is what these
#: attackers want; the others exist so the model has somewhere to put
#: surfaces that are not the user's prompt, rather than guessing.
ROLE_CATEGORIES: tuple[str, ...] = (
    "user-prompt",
    "content-injection",
    "system-prompt",
)

#: The role these optimizers deliver into.
USER_PROMPT_ROLE = "user-prompt"

#: Roles that must never receive a user-facing jailbreak prompt.
_NON_DELIVERY_ROLES = frozenset({"system-prompt"})

#: Backstop: the reserved name of the system prompt, never injected into, so a
#: Task-configured system prompt survives.
SYSTEM_PROMPT_CONTROLLABLE_NAME = "system_prompt"

#: Backstop: value types that carry free-form prose. A json/number-typed
#: surface would discard a prose payload.
FREE_TEXT_VALUE_TYPES = frozenset({"", "text", "str", "string", "html", "markdown"})


def is_eligible_surface(controllable: Controllable) -> bool:
    """Name/value-type backstop: could this surface carry a prose payload?"""
    if controllable.name == SYSTEM_PROMPT_CONTROLLABLE_NAME:
        return False
    return controllable.value_type.lower() in FREE_TEXT_VALUE_TYPES


def preferred_surface_names(
    controllables: Sequence[Controllable],
    roles: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Names of the surfaces the payload should be delivered into.

    Consults the attacker's LLM classification first: if any eligible surface
    was classified ``user-prompt``, only those are preferred, so the payload is
    not spent on an environment/content surface that merely happened to fire
    first. Falls back to every eligible surface when classification is
    unavailable or names none -- identical to the previous behaviour.
    """
    eligible = [c for c in controllables if is_eligible_surface(c)]
    if roles:
        chosen = {
            c.name
            for c in eligible
            if roles.get(c.name) == USER_PROMPT_ROLE
        }
        if chosen:
            return frozenset(chosen)
    return frozenset(c.name for c in eligible)


def should_inject(
    controllable: Controllable,
    preferred: frozenset[str],
    roles: Mapping[str, str] | None = None,
) -> bool:
    """Whether this surface should receive the payload.

    A target may raise a surface at run time that it never listed in
    ``get_controllables()``, so such a surface is unknown to both ``preferred``
    and ``roles``. Declining it would silently drop the attack, so an unknown
    surface falls back to the eligibility backstop; only a surface the
    classifier actually saw and did *not* prefer is skipped.
    """
    if not is_eligible_surface(controllable):
        return False
    if roles and roles.get(controllable.name) in _NON_DELIVERY_ROLES:
        return False
    if not preferred:
        return True
    if controllable.name in preferred:
        return True
    if roles and controllable.name in roles:
        return False  # classified, and something better exists
    return True  # never seen at initialize -> backstop


__all__ = [
    "FREE_TEXT_VALUE_TYPES",
    "ROLE_CATEGORIES",
    "SYSTEM_PROMPT_CONTROLLABLE_NAME",
    "USER_PROMPT_ROLE",
    "is_eligible_surface",
    "preferred_surface_names",
    "should_inject",
]
