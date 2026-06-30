"""Vessel selection: ranking a target's controllables as injection surfaces.

A *vessel* is a candidate injection surface, i.e. a single ``Controllable`` the
optimizer may inject into. MUZZLE prefers high-salience CONTENT surfaces -- the
``ControllablePostCallEvent`` surfaces that carry tool outputs and retrieved
documents, the indirect-injection channel the upstream attack assumes -- and
falls back to prompt surfaces (the user prompt first, then the system prompt)
only when no content surface is in scope.

The content / user-prompt surface predicates are reused verbatim from the
AgentVigil/WebSentinel optimizer (``agentvigil_websentinel_optimizer.optimizer``:
``_is_agent_content_surface`` / ``_is_content_postcall_surface`` / ``_is_user_prompt``)
so the two indirect-injection optimizers classify a target's surfaces
identically. ``build_vessels`` layers MUZZLE's grafter-guided ranking on top:
when the grafter LLM has proposed candidate web-UI anchors, content surfaces
named by a higher-confidence candidate are ranked first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from superred.core.types.controllable import Controllable

# The controllable name a target uses for its system prompt (last-resort vessel).
SYSTEM_PROMPT_NAME = "system_prompt"

# --- AgentVigil surface-classification constants (ported verbatim) -----------
# Reserved user-prompt controllable names (exact, case-insensitive match).
_USER_PROMPT_NAMES = {"user_prompt", "user_message", "query", "prompt"}
# ``read__<tool>`` PostCall content (e.g. AgentDojo read tools).
_TOOL_PREFIX = "read__"
# Per-tool OUTPUT (indirect-prompt-injection) surfaces, e.g. AgentHarm's
# inspect_agent target names them ``tool:<tool_name>`` ("replace the value the
# tool returns to the agent"). Distinct from ``tool_call:`` (the attacker's own
# tool): ``tool:`` has a colon at index 4, ``tool_call:`` an underscore, so they
# never prefix-collide.
_TOOL_OUTPUT_PREFIX = "tool:"
_ATTACKER_TOOL_PREFIX = "tool_call:"
_RESPONSE_NAME_HINTS = ("response", "assistant", "reply", "output", "completion")
_CONTENT_RESPONSE_NAME_HINTS = (
    "tool_response",
    "tool_result",
    "tool_output",
    "tool_return",
    "mcp_response",
    "mcp_result",
    "retrieval_response",
    "retrieval_result",
    "rag_response",
    "rag_result",
)
_CONTENT_SURFACE_HINTS = (
    "browser",
    "content",
    "context",
    "document",
    "external",
    "html",
    "knowledge",
    "mcp",
    "memory",
    "observation",
    "page",
    "rag",
    "read",
    "retrieval",
    "retrieved",
    "result",
    "tool_response",
    "web",
)
# User-prompt role tokens that, combined with "user", mark a prompt surface.
_USER_PROMPT_ROLE_HINTS = ("message", "prompt", "query", "task", "instruction")


def _is_content_postcall_surface(controllable: Controllable, extra_names: frozenset[str]) -> bool:
    """AgentVigil ``_is_content_postcall_surface`` with explicit ``extra_names``.

    ``extra_names`` replaces AgentVigil's instance-stored
    ``_content_controllable_names`` set; it is matched both as given and
    lower-cased so opaque per-target content names can be force-classified.
    """
    normalized = controllable.name.lower()
    known = extra_names | frozenset(name.lower() for name in extra_names)
    if controllable.name in known or normalized in known:
        return True
    metadata = f"{controllable.description} {controllable.value_type}".lower()
    metadata_looks_content_like = any(hint in metadata for hint in _CONTENT_SURFACE_HINTS)
    if any(hint in normalized for hint in _CONTENT_RESPONSE_NAME_HINTS):
        return True
    if any(hint in normalized for hint in _RESPONSE_NAME_HINTS) and not metadata_looks_content_like:
        return False
    haystack = " ".join(
        (controllable.name, controllable.description, controllable.value_type)
    ).lower()
    return any(hint in haystack for hint in _CONTENT_SURFACE_HINTS)


def is_content_surface(
    controllable: Controllable, *, extra_names: frozenset[str] = frozenset()
) -> bool:
    """True if ``controllable`` is a high-salience content surface.

    Verbatim port of AgentVigil ``_is_agent_content_surface``: ``read__`` /
    ``tool:`` / ``tool_call:`` prefixes, or a content-like PostCall surface.
    """
    name = controllable.name
    return (
        name.startswith(_TOOL_PREFIX)
        or name.startswith(_TOOL_OUTPUT_PREFIX)
        or name.startswith(_ATTACKER_TOOL_PREFIX)
        or _is_content_postcall_surface(controllable, extra_names)
    )


def is_user_prompt_surface(controllable: Controllable) -> bool:
    """True if ``controllable`` is the agent's user-prompt surface.

    Verbatim port of AgentVigil ``_is_user_prompt``.
    """
    normalized = controllable.name.lower()
    if normalized in _USER_PROMPT_NAMES:
        return True
    haystack = " ".join(
        (controllable.name, controllable.description, controllable.value_type)
    ).lower()
    has_user = "user" in haystack
    has_prompt_role = any(hint in haystack for hint in _USER_PROMPT_ROLE_HINTS)
    return has_user and has_prompt_role


@dataclass
class Vessel:
    """A ranked candidate injection surface.

    Attributes:
        controllable: The injection point this vessel wraps.
        kind: Surface class -- ``"content"``, ``"user_prompt"`` or
            ``"system_prompt"``.
        confidence: Grafter confidence in ``[0, 1]`` (``0.0`` when the surface
            was not named by any grafter candidate).
        element: Grafter natural-language description of the anchored web-UI
            element (empty when not grafter-named).
        why: Grafter rationale for the surface (empty when not grafter-named).
        how: Grafter scenario for placing content there (empty when not
            grafter-named).
    """

    controllable: Controllable
    kind: str
    confidence: float = 0.0
    element: str = ""
    why: str = ""
    how: str = ""


def _candidate_text(candidate: dict[str, Any]) -> str:
    """Lower-cased concatenation of a grafter candidate's free-text fields."""
    parts = (
        str(candidate.get("element", "")),
        str(candidate.get("how", "")),
        str(candidate.get("why", "")),
    )
    return " ".join(parts).lower()


def _safe_confidence(candidate: dict[str, Any]) -> float:
    """``float(candidate["confidence"])`` with a ``0.0`` fallback on bad values."""
    try:
        return float(candidate.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def build_vessels(
    controllables: list[Controllable],
    *,
    grafter_candidates: list[dict[str, Any]] | None = None,
    extra_content_names: frozenset[str] = frozenset(),
) -> list[Vessel]:
    """Rank ``controllables`` into a prioritized list of injection vessels.

    Order: content surfaces first (grafter-named ones ranked by confidence
    descending, then any remaining content surfaces at ``0.0`` in stable order),
    then a single user-prompt fallback, then a single system-prompt fallback.
    Each controllable appears at most once (deduped by object identity). A
    grafter candidate that names no in-scope content surface contributes
    nothing; when the grafter named none but content surfaces exist, they are
    all included at ``0.0``. Never raises on opaque or empty inputs.
    """
    content_controllables = [
        ctrl for ctrl in controllables if is_content_surface(ctrl, extra_names=extra_content_names)
    ]

    # Grafter-named content surfaces. A candidate "names" a content surface when
    # the controllable's name appears (case-insensitively) anywhere in the
    # candidate's element/how/why text -- this subsumes "a content-surface whose
    # name appears in the element text" since the element field is part of it.
    matched: list[Vessel] = []
    if grafter_candidates:
        for candidate in grafter_candidates:
            if not isinstance(candidate, dict):
                continue
            text = _candidate_text(candidate)
            confidence = _safe_confidence(candidate)
            element = str(candidate.get("element", ""))
            why = str(candidate.get("why", ""))
            how = str(candidate.get("how", ""))
            for ctrl in content_controllables:
                if ctrl.name and ctrl.name.lower() in text:
                    matched.append(
                        Vessel(
                            controllable=ctrl,
                            kind="content",
                            confidence=confidence,
                            element=element,
                            why=why,
                            how=how,
                        )
                    )

    # Stable sort by confidence DESC: the highest-confidence candidate wins the
    # per-controllable dedup below (ties keep candidate then controllable order).
    matched.sort(key=lambda vessel: vessel.confidence, reverse=True)

    vessels: list[Vessel] = []
    seen: set[int] = set()
    for vessel in matched:
        marker = id(vessel.controllable)
        if marker in seen:
            continue
        seen.add(marker)
        vessels.append(vessel)

    # Remaining in-scope content surfaces the grafter did not name (confidence 0.0).
    for ctrl in content_controllables:
        marker = id(ctrl)
        if marker in seen:
            continue
        seen.add(marker)
        vessels.append(Vessel(controllable=ctrl, kind="content"))

    # Single user-prompt fallback, deprioritized below every content surface.
    for ctrl in controllables:
        if id(ctrl) in seen:
            continue
        if is_user_prompt_surface(ctrl):
            seen.add(id(ctrl))
            vessels.append(Vessel(controllable=ctrl, kind="user_prompt"))
            break

    # Single system-prompt fallback (last resort).
    for ctrl in controllables:
        if id(ctrl) in seen:
            continue
        if ctrl.name == SYSTEM_PROMPT_NAME:
            seen.add(id(ctrl))
            vessels.append(Vessel(controllable=ctrl, kind="system_prompt"))
            break

    return vessels


__all__ = [
    "SYSTEM_PROMPT_NAME",
    "Vessel",
    "build_vessels",
    "is_content_surface",
    "is_user_prompt_surface",
]
