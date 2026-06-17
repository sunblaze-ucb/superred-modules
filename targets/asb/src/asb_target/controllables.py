"""Controllable injection points exposed by :class:`AsbTarget`.

One Controllable per ASB injection method, each tagged to its trust-boundary
surface so the attacker's *scope* selects the method:

- ``dpi_user_prompt`` (user)                  -> DPI (tamper the user task)
- ``opi_tool_observation`` (tools.<tool>)     -> OPI (tamper a tool return);
  fired per non-final tool return tagged to the firing tool's leaf, so any
  ``{tools}`` or single-tool scope sees an OPI surface.
- ``pot_system_demonstration`` (system_prompt) -> PoT backdoor
- ``mp_retrieved_workflow`` (memory)          -> Memory poisoning (read side)

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
    """An OPI Controllable for a tool whose observation is tagged *tag*.

    Built per firing tool (or per tool leaf for advertisement). The injected
    value is appended to that tool's returned observation
    (ASB: function_response += '; ' + value), on non-final steps.
    """
    return Controllable(
        name="opi_tool_observation",
        security_domain=tag,
        description=(
            "Observation Prompt Injection: text appended to a tool's returned "
            "observation on non-final steps (ASB: function_response += '; ' + value)."
        ),
        value_type="text",
    )


#: Full advertised surface: DPI, PoT, MP, plus one OPI Controllable per tool
#: leaf (so any tools.* scope, whole-ecosystem / scenario / single-tool, sees
#: an OPI surface).
CONTROLLABLES: list[Controllable] = [
    DPI_USER_PROMPT_CTRL,
    POT_SYSTEM_DEMONSTRATION_CTRL,
    MP_RETRIEVED_WORKFLOW_CTRL,
    *(opi_tool_observation_ctrl(tag) for tag in TOOL_OBSERVATION_TAGS.values()),
]

__all__ = [
    "DPI_USER_PROMPT_CTRL",
    "POT_SYSTEM_DEMONSTRATION_CTRL",
    "MP_RETRIEVED_WORKFLOW_CTRL",
    "opi_tool_observation_ctrl",
    "CONTROLLABLES",
]
