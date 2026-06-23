"""Controllable injection points exposed by :class:`AsbTarget`.

One Controllable per ASB injection method, each tagged to its trust-boundary
surface so the attacker's *scope* selects the method:

- ``dpi_user_prompt`` (user)                  -> DPI (tamper the user task)
- ``opi_tool_observation`` (tools.<tool>)     -> OPI (tamper a tool return);
  fired per non-final tool return tagged to the firing tool's leaf, so any
  ``{tools}`` or single-tool scope sees an OPI surface.
- ``pot_system_demonstration`` (system_prompt) -> PoT backdoor
- ``mp_retrieved_workflow`` (memory)          -> Memory poisoning (read side)
- ``tool_catalog_register`` (tool_catalogue_add)     -> add a new tool
- ``tool_catalog_replace`` (tool_catalogue_edit)     -> shadow a tool's behavior
- ``tool_catalog_rewrite_doc`` (tool_catalogue_edit) -> edit a tool's description
- ``tool_catalog_unregister`` (tool_catalogue_remove) -> remove a tool
  The four catalogue-edit controllables fire ONCE before planning. An added or
  behavior-replaced tool is SHADOWED: at call time its return comes from a
  per-call event tagged at the add/edit capability (so editing scope grants the
  call even when the tool's ``tools.*`` leaf is out of scope), and the original
  tool is not run. See :func:`tool_catalogue_call_ctrl`.

The target performs NO injection by default: with no attacker every site
declines and the run is a clean, upstream-faithful baseline. The
Controllables only *expose* the opportunity; the actual injected content is
an attacker's concern.
"""

from __future__ import annotations

from superred.core.types.controllable import Controllable
from superred.core.types.security_domain import SecurityDomainTag

from asb_target.security_tags import (
    MEMORY_TAG,
    SYSTEM_PROMPT_TAG,
    TOOL_CATALOGUE_ADD_TAG,
    TOOL_CATALOGUE_EDIT_TAG,
    TOOL_CATALOGUE_REMOVE_TAG,
    TOOL_OBSERVATION_TAGS,
    USER_TAG,
)

DPI_USER_PROMPT_CTRL = Controllable(
    name="dpi_user_prompt",
    security_domain=USER_TAG,
    description=(
        "Direct Prompt Injection: text appended to the benign user "
        "instruction before planning (ASB: task_input += value)."
    ),
    value_type="text",
)

POT_SYSTEM_DEMONSTRATION_CTRL = Controllable(
    name="pot_system_demonstration",
    security_domain=SYSTEM_PROMPT_TAG,
    description=(
        'Plan-of-Thought backdoor. JSON object {"trigger": str, '
        '"demonstration": str}: the demonstration block is appended after the '
        "target's always-present plan-format scaffolding in the system "
        "instruction, and the trigger is appended to the user task so the "
        "backdoor fires (ASB build_system_instruction + task trigger)."
    ),
    value_type="json",
)

MP_RETRIEVED_WORKFLOW_CTRL = Controllable(
    name="mp_retrieved_workflow",
    security_domain=MEMORY_TAG,
    description=(
        "Memory poisoning (read side). At the memory-read point the agent "
        "retrieves the top-1 record from its durable store; the default is "
        "the genuine retrieved record, and an injection here substitutes the "
        "record the agent is told to follow (ASB search_memory_instruction)."
    ),
    value_type="text",
)


def opi_tool_observation_ctrl(tag: SecurityDomainTag) -> Controllable:
    """An OPI Controllable for a tool whose interaction is tagged *tag*.

    Built per firing tool (or per tool leaf for advertisement). OPI tampers the
    tool's RETURNED observation, so it is delivered as a POST-call event: the
    ``answer`` carries the genuine observation, the ``request`` carries the call
    ({tool, params}), and the injected value is appended to that tool's returned
    observation (ASB: function_response += '; ' + value), on non-final steps.
    """
    return Controllable(
        name="opi_tool_observation",
        security_domain=tag,
        description=(
            "Observation Prompt Injection. A post-call event: its answer carries "
            "the tool's returned observation and its request carries the call "
            "{tool, params}. The injected text is appended to that tool's returned "
            "observation on non-final steps (ASB: function_response += '; ' + value)."
        ),
        value_type="text",
    )


# ---------------------------------------------------------------------------
# Tool-catalogue editing (the REGISTRY). A superred extension beyond upstream
# ASB's single fixed attacker tool (see ASSUMPTIONS.md): the attacker may add /
# replace / rewrite-doc / unregister tools in the agent's catalogue, fired ONCE
# before planning. Names + the add/edit/remove capability split mirror the
# agentdojo and inspect-agent targets so a cross-target optimizer's catalogue
# handlers work here too. A REPLACED or ADDED tool is SHADOWED: when the agent
# calls it the attacker supplies the return at call time (see
# tool_catalogue_call_ctrl) and the original tool is NOT run. A rewrite-doc edit
# only changes the description the model reads (influencing selection), leaving
# behavior intact.
# ---------------------------------------------------------------------------

TOOL_CATALOG_REGISTER_CTRL = Controllable(
    name="tool_catalog_register",
    security_domain=TOOL_CATALOGUE_ADD_TAG,
    description=(
        "Add a NEW attacker-defined tool to the agent's catalogue (fired once "
        'before planning). JSON object {"name": str, "description": str, '
        '"fake_return": str}: the tool appears in the catalogue the model plans '
        "over; when the agent calls it the attacker supplies its return (per-call "
        "event, or the static fake_return), and no original tool runs."
    ),
    value_type="json",
)

TOOL_CATALOG_REPLACE_CTRL = Controllable(
    name="tool_catalog_replace",
    security_domain=TOOL_CATALOGUE_EDIT_TAG,
    description=(
        "Replace (shadow) an EXISTING tool's behavior (fired once before "
        'planning). JSON object {"name": str, "fake_return": str, '
        '"description"?: str}: when the agent calls the named tool the attacker '
        "supplies its return (per-call event, or the static fake_return) and the "
        "original tool is NOT run; an optional description also updates the "
        "listing the model reads."
    ),
    value_type="json",
)

TOOL_CATALOG_UNREGISTER_CTRL = Controllable(
    name="tool_catalog_unregister",
    security_domain=TOOL_CATALOGUE_REMOVE_TAG,
    description=(
        "Remove an existing tool from the catalogue (fired once before "
        'planning). JSON object {"name": str}: the tool is dropped so the agent '
        "can no longer select or call it."
    ),
    value_type="json",
)

TOOL_CATALOG_REWRITE_DOC_CTRL = Controllable(
    name="tool_catalog_rewrite_doc",
    security_domain=TOOL_CATALOGUE_EDIT_TAG,
    description=(
        "Rewrite an existing tool's DESCRIPTION without changing its behavior "
        '(fired once before planning). JSON object {"name": str, '
        '"description": str}: only the catalogue listing the model reads when '
        "choosing tools changes; the original tool still runs when called."
    ),
    value_type="json",
)

TOOL_CATALOG_EDIT_CTRLS: tuple[Controllable, ...] = (
    TOOL_CATALOG_REGISTER_CTRL,
    TOOL_CATALOG_REPLACE_CTRL,
    TOOL_CATALOG_UNREGISTER_CTRL,
    TOOL_CATALOG_REWRITE_DOC_CTRL,
)


def tool_catalogue_call_ctrl(tag: SecurityDomainTag, tool_name: str) -> Controllable:
    """A per-call Controllable for an attacker-managed (added or behavior-replaced)
    tool, tagged at the ADD/EDIT capability *tag* -- NOT the tool's ``tools.*``
    leaf -- so an attacker holding only the catalogue add/edit capability
    receives and controls the call even when the tool's own leaf is out of scope.

    The event ``request`` carries the call {tool, params}; a
    ``ControllableInjection`` supplies the tool's return directly. The original
    tool is never run.
    """
    return Controllable(
        name=f"tool_call:{tool_name}",
        security_domain=tag,
        description=(
            "Per-call event for an attacker-added or behavior-replaced tool "
            f"({tool_name!r}). The event request carries {{tool, params}}; a "
            "ControllableInjection supplies the tool's return directly, without "
            "running any original tool."
        ),
        value_type="text",
    )


#: Full advertised surface: DPI, PoT, MP, the four tool-catalogue edit
#: controllables, plus one OPI Controllable per tool leaf (so any tools.* scope,
#: whole-ecosystem / scenario / single-tool, sees an OPI surface).
CONTROLLABLES: list[Controllable] = [
    DPI_USER_PROMPT_CTRL,
    POT_SYSTEM_DEMONSTRATION_CTRL,
    MP_RETRIEVED_WORKFLOW_CTRL,
    *TOOL_CATALOG_EDIT_CTRLS,
    *(opi_tool_observation_ctrl(tag) for tag in TOOL_OBSERVATION_TAGS.values()),
]

__all__ = [
    "DPI_USER_PROMPT_CTRL",
    "POT_SYSTEM_DEMONSTRATION_CTRL",
    "MP_RETRIEVED_WORKFLOW_CTRL",
    "opi_tool_observation_ctrl",
    "TOOL_CATALOG_REGISTER_CTRL",
    "TOOL_CATALOG_REPLACE_CTRL",
    "TOOL_CATALOG_UNREGISTER_CTRL",
    "TOOL_CATALOG_REWRITE_DOC_CTRL",
    "TOOL_CATALOG_EDIT_CTRLS",
    "tool_catalogue_call_ctrl",
    "CONTROLLABLES",
]
