"""Observable specs and helper builders exposed by :class:`AgentDojoTarget`.

Two flavours:

- **Static** observables: returned from
  :meth:`Target.get_observables` and re-built per call so the
  ``content`` reflects the target's *current* configuration (system
  prompt, tool catalog snapshot, model identity, etc.).  These are
  available to the optimizer in ``initialize()`` before any run starts.

- **Dynamic** observables: emitted at runtime via ``emit(ObservableEvent(...))``
  inside :meth:`Target.run` (and the pipeline hooks / wrapped runtime).
  These cover the agent trace (one event per chat message, per agent
  tool call, and per agent-visible tool return), one event per mutating
  (write) tool call tagged at the store it changed, and three internal
  control events (tool-menu rebuild, a discarded malformed action, and
  the outcome of each attacker catalogue edit).  Per-read legitimate
  values are NOT mirrored as observables: each is carried once, on its
  per-read ``ControllablePostCallEvent``; a Controller that lists the
  store leaf under ``read_only`` observes those events from the
  trajectory without being able to inject.  No observable mirrors store
  contents reachable through a read controllable; the full environment
  state is available only post-run to the scorer via the query specs.

For dynamic observables, a builder function in this module produces the
:class:`Observable` spec; the caller attaches the per-event content via
:class:`ObservableEvent`.
"""

from __future__ import annotations

from superred.core.types.observable import Observable
from superred.core.types.security_domain import SecurityDomainTag

from agentdojo_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    AGENT_TRACE_TOOL_RESPONSES_TAG,
    MODEL_IDENTITY_TAG,
    TOOL_CATALOGUE_TAG,
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

STATIC_OBSERVABLE_SPECS: tuple[Observable, ...] = (
    MODEL_IDENTITY_OBS,
    TOOL_CATALOG_LISTING_OBS,
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
    """Observable for one tool return value (post-injection, as the agent saw it).

    The content is ``{"function": <suite-prefixed tool name>, "value": ...,
    "error": ...}``: the producing tool's name travels in the same record as
    its output, so a single response is self-identifying without
    cross-referencing the call observable by index.
    """
    return Observable(
        name=f"agent_trace_tool_response_{call_index:04d}",
        security_domain=AGENT_TRACE_TOOL_RESPONSES_TAG,
        description=(
            "The return value the agent observed for tool call "
            f"{call_index} (after any on-demand injection was applied), with "
            "the producing tool's suite-prefixed name under 'function'."
        ),
        observable_type="json",
    )


def write_observation_observable(
    call_index: int, store_tag: SecurityDomainTag
) -> Observable:
    """Observable for one mutating (write) tool call, tagged at the store
    it changed.

    Lets a service-scoped attacker see the action it provoked under the
    same boundary it reads from (reading and acting on a store share a
    label).  ``store_tag`` is the write tool's entry in
    :data:`agentdojo_target.controllables.WRITE_STORE_MAP`.
    """
    return Observable(
        name=f"write_call_{call_index:04d}",
        security_domain=store_tag,
        description=(
            "A mutating tool call the agent performed, observed under the "
            f"store it changed (position {call_index})."
        ),
        observable_type="json",
    )


def tool_menu_rebuild_observable(rebuild_index: int) -> Observable:
    """Observable for the tool menu being rebuilt mid-run.

    Emitted when the wrapped runtime re-syncs its function registry after
    an attacker catalogue edit, so a trace-scoped reader sees that the
    menu the agent is offered changed."""
    return Observable(
        name=f"tool_menu_rebuild_{rebuild_index:04d}",
        security_domain=AGENT_TRACE_TAG,
        description=(
            f"The agent's tool menu was rebuilt mid-run (rebuild {rebuild_index})."
        ),
        observable_type="json",
    )


def discarded_action_observable(discard_index: int) -> Observable:
    """Observable for a malformed tool call the model requested being
    discarded.

    Emitted when the compatibility layer drops a pattern-violating
    tool-call name from an assistant response, so a reader sees the
    action was attempted and dropped rather than silently vanishing."""
    return Observable(
        name=f"discarded_action_{discard_index:04d}",
        security_domain=AGENT_TRACE_TOOL_CALLS_TAG,
        description=(
            "A malformed tool call the model requested was discarded "
            f"(discard {discard_index})."
        ),
        observable_type="json",
    )


def catalog_edit_outcome_observable(
    operation: str, security_domain: SecurityDomainTag
) -> Observable:
    """Observable for the outcome of one attacker tool-catalogue edit.

    Records whether a register / replace / unregister / rewrite-doc edit
    was applied or rejected (and why), tagged at the catalogue boundary
    that authorises the edit (register at the addable sub-boundary, the
    others at the broader catalogue boundary)."""
    return Observable(
        name=f"catalog_edit_{operation}_outcome",
        security_domain=security_domain,
        description=(
            f"Outcome of the attacker's '{operation}' tool-catalogue edit "
            "(applied or rejected)."
        ),
        observable_type="json",
    )


__all__ = [
    # static specs
    "MODEL_IDENTITY_OBS",
    "TOOL_CATALOG_LISTING_OBS",
    "STATIC_OBSERVABLE_SPECS",
    # dynamic builders
    "chat_message_observable",
    "agent_tool_call_observable",
    "agent_tool_response_observable",
    "write_observation_observable",
    "tool_menu_rebuild_observable",
    "discarded_action_observable",
    "catalog_edit_outcome_observable",
]
