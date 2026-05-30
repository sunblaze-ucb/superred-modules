"""Observable specs and helper builders exposed by :class:`AgentDojoTarget`.

Two flavours:

- **Static** observables: returned from
  :meth:`Target.get_observables` and re-built per call so the
  ``content`` reflects the target's *current* configuration (system
  prompt, tool catalog snapshot, model identity, etc.).  These are
  available to the optimizer in ``initialize()`` before any run starts.

- **Dynamic** observables: emitted at runtime via ``emit(ObservableEvent(...))``
  inside :meth:`Target.run`.  These cover the agent trace (one event
  per chat message, per agent tool call, per agent-visible tool return),
  per-read legitimate data fields (so an optimizer with the right
  ``tools.<leaf>`` scope can read the pre-injection value via the
  observable channel as well), and per write-side tool call (for Layer-2
  predicates that need to detect agent mutations).

For dynamic observables, a builder function in this module produces the
:class:`Observable` spec; the caller (:class:`AgentDojoTarget`) attaches
the per-event content via :class:`ObservableEvent`.
"""

from __future__ import annotations

from superred.core.types.observable import Observable

from agentdojo_target.controllables import READ_CTRLS
from agentdojo_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    AGENT_TRACE_TOOL_RESPONSES_TAG,
    MODEL_IDENTITY_TAG,
    PROMPT_READABLE_TAG,
    TOOL_CATALOGUE_READABLE_TAG,
    TOOLS_TAG,
)

# ---------------------------------------------------------------------------
# Static observable specs (the specs are stable; content is rebuilt per run)
# ---------------------------------------------------------------------------

MODEL_IDENTITY_OBS: Observable = Observable(
    name="model_identity",
    security_domain=MODEL_IDENTITY_TAG,
    description="The litellm model id powering the underlying AgentDojo pipeline.",
    observable_type="text",
)

SYSTEM_PROMPT_OBS: Observable = Observable(
    name="system_prompt",
    security_domain=PROMPT_READABLE_TAG,
    description="The agent's current system prompt text.",
    observable_type="text",
)

TOOL_CATALOG_LISTING_OBS: Observable = Observable(
    name="tool_catalog_listing",
    security_domain=TOOL_CATALOGUE_READABLE_TAG,
    description=(
        "JSON snapshot of the currently-registered tool catalog: each "
        "entry is {name, description, parameters_schema}."
    ),
    observable_type="json",
)

COMPOSITE_ENV_SNAPSHOT_OBS: Observable = Observable(
    name="composite_env_snapshot",
    security_domain=TOOLS_TAG,
    description=(
        "JSON dump of the full CompositeEnvironment (banking, workspace, "
        "slack, travel sub-envs).  Emitted at run end."
    ),
    observable_type="json",
)

STATIC_OBSERVABLE_SPECS: tuple[Observable, ...] = (
    MODEL_IDENTITY_OBS,
    SYSTEM_PROMPT_OBS,
    TOOL_CATALOG_LISTING_OBS,
    COMPOSITE_ENV_SNAPSHOT_OBS,
)

# ---------------------------------------------------------------------------
# Dynamic observable spec builders (one Observable per occurrence)
# ---------------------------------------------------------------------------


def chat_message_observable(message_index: int) -> Observable:
    """Observable for a single chat message at position *message_index*.

    The index is recorded in the name so consumers can correlate events
    with positions in the message stream.
    """
    return Observable(
        name=f"agent_trace_message_{message_index:04d}",
        security_domain=AGENT_TRACE_MESSAGES_TAG,
        description=(
            "One chat message from the agent pipeline's run "
            f"(position {message_index})."
        ),
        observable_type="json",
    )


def agent_tool_call_observable(call_index: int) -> Observable:
    """Observable for a single FunctionCall the agent emits."""
    return Observable(
        name=f"agent_trace_tool_call_{call_index:04d}",
        security_domain=AGENT_TRACE_TOOL_CALLS_TAG,
        description=(
            "One FunctionCall the agent attempted "
            f"(position {call_index} in the trace)."
        ),
        observable_type="json",
    )


def agent_tool_response_observable(call_index: int) -> Observable:
    """Observable for one tool return value (post-injection, as the agent saw it)."""
    return Observable(
        name=f"agent_trace_tool_response_{call_index:04d}",
        security_domain=AGENT_TRACE_TOOL_RESPONSES_TAG,
        description=(
            "The return value the agent observed for tool call "
            f"{call_index} (after any on-demand injection was applied)."
        ),
        observable_type="json",
    )


def write_call_observable(tool_prefixed_name: str) -> Observable:
    """Observable emitted on each write-side tool invocation.

    Carries the function name and args via the
    :class:`ObservableEvent.content` payload so Layer-2 predicates can
    scan the trajectory for unauthorized agent actions even when the
    post-environment diff is ambiguous (e.g. when the action was a no-op
    on the environment but still an attempted side effect).
    """
    return Observable(
        name=f"write_call:{tool_prefixed_name}",
        security_domain=AGENT_TRACE_TOOL_CALLS_TAG,
        description=(
            f"Write-side invocation of ``{tool_prefixed_name}`` recorded "
            "by the runtime wrapper.  Content carries {function, args}."
        ),
        observable_type="json",
    )


def read_data_field_observable(tool_prefixed_name: str) -> Observable:
    """Observable mirror of a per-read controllable's *legitimate* value.

    The corresponding Controllable's ``security_domain`` is reused so
    that an optimizer whose scope includes the per-read tag sees BOTH
    the controllable event (where it can inject) AND the observable
    event (where it sees the pre-injection legitimate value).  An
    optimizer whose scope includes only ``tools.<leaf>`` via the
    *observable* channel sees the legitimate value without being able
    to inject.
    """
    ctrl = READ_CTRLS.get(tool_prefixed_name)
    if ctrl is None:
        raise KeyError(
            f"No read Controllable registered for {tool_prefixed_name!r}; "
            "cannot build a mirroring observable."
        )
    return Observable(
        name=f"read_data_field:{tool_prefixed_name}",
        security_domain=ctrl.security_domain,
        description=(
            f"Legitimate pre-injection return value for read tool "
            f"``{tool_prefixed_name}``."
        ),
        observable_type="json",
    )


__all__ = [
    # static specs
    "MODEL_IDENTITY_OBS",
    "SYSTEM_PROMPT_OBS",
    "TOOL_CATALOG_LISTING_OBS",
    "COMPOSITE_ENV_SNAPSHOT_OBS",
    "STATIC_OBSERVABLE_SPECS",
    # dynamic builders
    "chat_message_observable",
    "agent_tool_call_observable",
    "agent_tool_response_observable",
    "write_call_observable",
    "read_data_field_observable",
]
