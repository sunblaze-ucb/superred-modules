"""Observable specs and dynamic builders for :class:`InspectAgentTarget`.

Convention (matches the target's design): STATIC configuration is exposed as
static observables (returned from ``get_observables`` with current content);
everything that HAPPENS during the run is emitted onto the trajectory as a
dynamic ``ObservableEvent``.

Static observables (``get_observables``):
- ``model_identity``: which model powers the agent.
- ``system_prompt``: the current system prompt text.
- ``message_limit``: the rollout's message cap.
- ``tool_catalog_listing``: the configured (pre-edit) tool catalogue.

Dynamic observables (emitted via ``emit(ObservableEvent(...))`` during ``run``):
- one ``agent_trace_message_NNNN`` per chat message in the rollout.
- one ``agent_trace_tool_call_NNNN`` per tool call the agent emits.
- one ``agent_trace_tool_response_NNNN`` per tool return the agent observes.
"""

from __future__ import annotations

from superred.core.types.observable import Observable

from inspect_agent_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    AGENT_TRACE_TOOL_RESPONSES_TAG,
    MESSAGE_LIMIT_READABLE_TAG,
    MODEL_IDENTITY_TAG,
    SYSTEM_PROMPT_READABLE_TAG,
    TOOL_CATALOGUE_READABLE_TAG,
)

MODEL_IDENTITY_OBS: Observable = Observable(
    name="model_identity",
    security_domain=MODEL_IDENTITY_TAG,
    description="The litellm-style model id powering the agent.",
    observable_type="text",
)

SYSTEM_PROMPT_OBS: Observable = Observable(
    name="system_prompt",
    security_domain=SYSTEM_PROMPT_READABLE_TAG,
    description="The agent's current system prompt text.",
    observable_type="text",
)

MESSAGE_LIMIT_OBS: Observable = Observable(
    name="message_limit",
    security_domain=MESSAGE_LIMIT_READABLE_TAG,
    description="The rollout's message-limit cap for this run (integer as text).",
    observable_type="text",
)

TOOL_CATALOG_LISTING_OBS: Observable = Observable(
    name="tool_catalog_listing",
    security_domain=TOOL_CATALOGUE_READABLE_TAG,
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
    SYSTEM_PROMPT_OBS,
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


def agent_tool_call_observable(call_index: int) -> Observable:
    """Observable for a single tool call the agent emits."""
    return Observable(
        name=f"agent_trace_tool_call_{call_index:04d}",
        security_domain=AGENT_TRACE_TOOL_CALLS_TAG,
        description=f"One tool call the agent attempted (position {call_index}).",
        observable_type="json",
    )


def agent_tool_response_observable(response_index: int) -> Observable:
    """Observable for a single tool return value the agent observed.

    Carries the output *after* any tool-output injection has been applied, so an
    optimizer scoped to read tool responses sees exactly what the agent saw.
    """
    return Observable(
        name=f"agent_trace_tool_response_{response_index:04d}",
        security_domain=AGENT_TRACE_TOOL_RESPONSES_TAG,
        description=f"One tool return value the agent observed (position {response_index}).",
        observable_type="json",
    )


__all__ = [
    "MODEL_IDENTITY_OBS",
    "SYSTEM_PROMPT_OBS",
    "MESSAGE_LIMIT_OBS",
    "TOOL_CATALOG_LISTING_OBS",
    "STATIC_OBSERVABLE_SPECS",
    "chat_message_observable",
    "agent_tool_call_observable",
    "agent_tool_response_observable",
]
