"""Observable specs and helper builders exposed by :class:`AgentDojoTarget`.

Two flavours:

- **Static** observables: returned from
  :meth:`Target.get_observables` and re-built per call so the
  ``content`` reflects the target's *current* configuration (system
  prompt, tool catalog snapshot, model identity, etc.).  These are
  available to the optimizer in ``initialize()`` before any run starts.

- **Dynamic** observables: emitted at runtime via ``emit(ObservableEvent(...))``
  inside :meth:`Target.run`.  These cover the agent trace: one event
  per chat message, per agent tool call, and per agent-visible tool
  return.  Per-read legitimate values are NOT mirrored here — each is
  carried exactly once, on its per-read ``ControllablePostCallEvent``;
  a Controller that lists the quadrant tag under ``read_only`` observes those
  events from the trajectory without being able to inject.

For dynamic observables, a builder function in this module produces the
:class:`Observable` spec; the caller (:class:`AgentDojoTarget`) attaches
the per-event content via :class:`ObservableEvent`.
"""

from __future__ import annotations

from superred.core.types.observable import Observable

from agentdojo_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    AGENT_TRACE_TOOL_RESPONSES_TAG,
    MODEL_IDENTITY_TAG,
    TOOL_CATALOGUE_TAG,
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

TOOL_CATALOG_LISTING_OBS: Observable = Observable(
    name="tool_catalog_listing",
    security_domain=TOOL_CATALOGUE_TAG,
    description=(
        "JSON snapshot of the seeded (pre-edit) tool catalog: each "
        "entry is {name, description, parameters_schema}.  Attacker "
        "edits appear on the trajectory as the catalogue controllable "
        "events."
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


__all__ = [
    # static specs
    "MODEL_IDENTITY_OBS",
    "TOOL_CATALOG_LISTING_OBS",
    "COMPOSITE_ENV_SNAPSHOT_OBS",
    "STATIC_OBSERVABLE_SPECS",
    # dynamic builders
    "chat_message_observable",
    "agent_tool_call_observable",
    "agent_tool_response_observable",
]
