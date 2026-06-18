"""Observable specs and builders exposed by :class:`AsbTarget`.

Static observables (read once at optimizer init, rebuilt per call so content
reflects the current configuration):

- ``system_prompt``       : the agent's current system-instruction persona.
- ``model_identity``      : the litellm model id powering the agent.
- ``tool_catalog_listing``: JSON of the agent's tool registry (incl. attacker
  tool), tagged at the ``tool_catalogue`` boundary (NOT the ``tools`` tree).

Dynamic observables (emitted during the run via ``emit(ObservableEvent(...))``)
record each genuinely-new runtime fact ONCE at its true provenance and in
causal order (no bulk message dump, no cross-zone leak):

- ``agent_plan``               : the planning workflow JSON the model produced
  (the agent's own generation, under ``agent_trace``).
- ``agent_model_output_NNNN``  : a per-step model output ([Thinking] text), the
  agent's own generation (under ``agent_trace``).
- ``tool_interaction_NNNN``    : a whole tool interaction (the call, its
  parameters, and the returned observation) on the FINAL step, tagged to the
  firing tool's own ``tools.*`` leaf -- a tool interaction is the tool's data,
  not the agent's. NON-final tool interactions are not observables: they ride
  the trajectory as their OPI controllable event (whose ``request`` carries the
  same {tool, params, observation}, and where the attacker can change the
  return), also under the tool. So each tool interaction is emitted exactly
  once, under its tool, and the tool call is never a separate ``agent_trace``
  record.
- ``memory_read`` / ``memory_write`` : a durable-memory read/write event.

The attack payload is NOT exposed (the target only exposes injection points;
attacks are an attacker concern). Model identity IS exposed as a read-only
observable (which model powers the agent is attacker-relevant, e.g. for
model-specific payloads) but is never a controllable (the model is a
construction concern).
"""

from __future__ import annotations

from superred.core.types.observable import Observable
from superred.core.types.security_domain import SecurityDomainTag

from asb_target.security_tags import (
    AGENT_TRACE_TAG,
    MEMORY_TAG,
    MODEL_IDENTITY_TAG,
    SYSTEM_PROMPT_TAG,
    TOOL_CATALOGUE_TAG,
)

# ---------------------------------------------------------------------------
# Static observable specs (content rebuilt per run)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_OBS = Observable(
    name="system_prompt",
    security_domain=SYSTEM_PROMPT_TAG,
    description="The agent's current system instruction (persona) text.",
    observable_type="text",
)

MODEL_IDENTITY_OBS = Observable(
    name="model_identity",
    security_domain=MODEL_IDENTITY_TAG,
    description="The litellm-style model id powering the agent.",
    observable_type="text",
)

TOOL_CATALOG_LISTING_OBS = Observable(
    name="tool_catalog_listing",
    security_domain=TOOL_CATALOGUE_TAG,
    description=(
        "JSON snapshot of the agent's tool catalogue (the registry: which tools "
        "exist + their descriptions): each entry is {name, description}, "
        "including the attacker tool once registered. Tagged at the "
        "tool_catalogue registry boundary, NOT the tools tree, so reading the "
        "listing does not grant reading any tool's returned observations."
    ),
    observable_type="json",
)

STATIC_OBSERVABLE_SPECS: tuple[Observable, ...] = (
    SYSTEM_PROMPT_OBS,
    MODEL_IDENTITY_OBS,
    TOOL_CATALOG_LISTING_OBS,
)

# ---------------------------------------------------------------------------
# Dynamic observable builders (one per occurrence; index in the name)
# ---------------------------------------------------------------------------


def agent_plan_observable() -> Observable:
    """The planning workflow JSON the model generated (emitted once per run)."""
    return Observable(
        name="agent_plan",
        security_domain=AGENT_TRACE_TAG,
        description="The plan-of-steps workflow JSON the agent generated.",
        observable_type="json",
    )


def agent_model_output_observable(step_index: int) -> Observable:
    """One per-step model output (the agent's [Thinking] text at *step_index*)."""
    return Observable(
        name=f"agent_model_output_{step_index:04d}",
        security_domain=AGENT_TRACE_TAG,
        description=f"The agent's model output at step {step_index}.",
        observable_type="text",
    )


def tool_interaction_observable(call_index: int, tag: SecurityDomainTag) -> Observable:
    """A whole tool interaction (call + parameters + returned observation) on the
    FINAL step, tagged to the firing tool's own boundary (*tag*, a ``tools.*``
    leaf), since a tool interaction is the tool's data. Non-final interactions
    are recorded via their OPI controllable event (also under the tool), so each
    tool interaction is emitted exactly once under its tool."""
    return Observable(
        name=f"tool_interaction_{call_index:04d}",
        security_domain=tag,
        description=(
            f"The final-step tool interaction at position {call_index}: "
            "{tool, params, observation}."
        ),
        observable_type="json",
    )


def memory_event_observable(kind: str) -> Observable:
    """A durable-memory event: *kind* is ``"read"`` or ``"write"``."""
    return Observable(
        name=f"memory_{kind}",
        security_domain=MEMORY_TAG,
        description=f"A durable-memory {kind} event (the record involved).",
        observable_type="text",
    )


__all__ = [
    "SYSTEM_PROMPT_OBS",
    "MODEL_IDENTITY_OBS",
    "TOOL_CATALOG_LISTING_OBS",
    "STATIC_OBSERVABLE_SPECS",
    "agent_plan_observable",
    "agent_model_output_observable",
    "tool_interaction_observable",
    "memory_event_observable",
]
