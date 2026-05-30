"""Observable specs and dynamic builders for :class:`InspectAgentTarget`.

Static observables (returned from ``get_observables`` with current content):
- ``model_identity``: which model powers the agent.
- ``system_prompt``: the current system prompt text.

Dynamic observables (emitted via ``emit(ObservableEvent(...))`` during ``run``):
- one ``agent_trace_message_NNNN`` per chat message in the rollout.
- one ``agent_trace_tool_call_NNNN`` per tool call the agent emits.
"""

from __future__ import annotations

from superred.core.types.observable import Observable

from inspect_agent_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    MODEL_IDENTITY_TAG,
    SYSTEM_PROMPT_READABLE_TAG,
    TOOLS_READABLE_TAG,
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

TOOL_CATALOG_LISTING_OBS: Observable = Observable(
    name="tool_catalog_listing",
    security_domain=TOOLS_READABLE_TAG,
    description=(
        "JSON snapshot of the current tool catalogue: a list of "
        "{name, description, parameters_schema} entries.  Re-emitted each turn "
        "so it reflects any attacker catalogue edits."
    ),
    observable_type="json",
)

STATIC_OBSERVABLE_SPECS: tuple[Observable, ...] = (
    MODEL_IDENTITY_OBS,
    SYSTEM_PROMPT_OBS,
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


__all__ = [
    "MODEL_IDENTITY_OBS",
    "SYSTEM_PROMPT_OBS",
    "TOOL_CATALOG_LISTING_OBS",
    "STATIC_OBSERVABLE_SPECS",
    "chat_message_observable",
    "agent_tool_call_observable",
]
