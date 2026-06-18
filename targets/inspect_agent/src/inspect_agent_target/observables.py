"""Observable specs and dynamic builders for :class:`InspectAgentTarget`.

Convention (matches the target's design): STATIC configuration is exposed as
static observables (returned from ``get_observables`` with current content);
everything that HAPPENS during the run is emitted onto the trajectory as a
dynamic ``ObservableEvent``.

Static observables (``get_observables``):
- ``model_identity``: which model powers the agent.
- ``message_limit``: the rollout's message cap.
- ``tool_catalog_listing``: the configured (pre-edit) tool catalogue.

The system prompt is NOT mirrored as an observable: it is carried exactly
once, on the system-prompt ``ControllablePreCallEvent`` at run start.  A
Controller that lists ``system_prompt`` under ``read_only`` reads it from
the trajectory without being able to override it.

Dynamic observables (emitted via ``emit(ObservableEvent(...))`` during ``run``):
- one ``agent_trace_message_NNNN`` per NON-TOOL chat message in the rollout
  (system / assistant reasoning / user).  Tool-call requests and tool-result
  messages are excluded: a tool's call (function + arguments) and its return are
  emitted exactly once, on that tool's ``ControllablePostCallEvent``.
"""

from __future__ import annotations

from superred.core.types.observable import Observable

from inspect_agent_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    DETAILED_SYSTEM_SPECIFICATION_TAG,
    MESSAGE_LIMIT_TAG,
    MODEL_IDENTITY_TAG,
    TOOL_CATALOGUE_TAG,
)

MODEL_IDENTITY_OBS: Observable = Observable(
    name="model_identity",
    security_domain=MODEL_IDENTITY_TAG,
    description="The litellm-style model id powering the agent.",
    observable_type="text",
)

DETAILED_SYSTEM_SPECIFICATION_OBS: Observable = Observable(
    name="detailed_system_specification",
    security_domain=DETAILED_SYSTEM_SPECIFICATION_TAG,
    description=(
        "A leaked, free-text design and threat-intelligence brief on this target "
        "system: its purpose, architecture, runtime logic, and hardcoded "
        "prompts/code, each at abstract / specific / examples depth.  Static "
        "intelligence the attacker can read to understand how the system works "
        "and craft stronger attacks; read-only, never a controllable."
    ),
    observable_type="text",
)

MESSAGE_LIMIT_OBS: Observable = Observable(
    name="message_limit",
    security_domain=MESSAGE_LIMIT_TAG,
    description="The rollout's message-limit cap for this run (integer as text).",
    observable_type="text",
)

TOOL_CATALOG_LISTING_OBS: Observable = Observable(
    name="tool_catalog_listing",
    security_domain=TOOL_CATALOGUE_TAG,
    description=(
        "JSON snapshot of the CONFIGURED (pre-edit) tool catalogue: a list of "
        "{name, description, parameters_schema} entries.  Static observable taken "
        "before any attacker catalogue edits; those edits appear on the trajectory "
        "as the tool-catalogue controllable events."
    ),
    observable_type="json",
)

STATIC_OBSERVABLE_SPECS: tuple[Observable, ...] = (
    MODEL_IDENTITY_OBS,
    DETAILED_SYSTEM_SPECIFICATION_OBS,
    MESSAGE_LIMIT_OBS,
    TOOL_CATALOG_LISTING_OBS,
)


def chat_message_observable(message_index: int) -> Observable:
    """Observable for a single chat message at position *message_index*."""
    return Observable(
        name=f"agent_trace_message_{message_index:04d}",
        security_domain=AGENT_TRACE_MESSAGES_TAG,
        description=f"One chat message from the agent rollout (position {message_index}).",
        observable_type="json",
    )


__all__ = [
    "MODEL_IDENTITY_OBS",
    "DETAILED_SYSTEM_SPECIFICATION_OBS",
    "MESSAGE_LIMIT_OBS",
    "TOOL_CATALOG_LISTING_OBS",
    "STATIC_OBSERVABLE_SPECS",
    "chat_message_observable",
]
