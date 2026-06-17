"""Observable specs and builders exposed by :class:`AsbTarget`.

Static observables (read once at optimizer init, rebuilt per call so content
reflects the current configuration):

- ``system_prompt``       : the agent's current system-instruction persona.
- ``tool_catalog_listing``: JSON of the agent's tools (incl. attacker tool).

Dynamic observables (emitted during the run via ``emit(ObservableEvent(...))``)
record each genuinely-new runtime fact ONCE at its true provenance and in
causal order (no bulk message dump, no cross-zone leak):

- ``agent_plan``               : the planning workflow JSON the model produced.
- ``agent_model_output_NNNN``  : a per-step model output ([Thinking] text).
- ``agent_tool_call_NNNN``     : an executed tool-call decision (the agent's
  own output, under ``agent_trace``).
- ``tool_response_NNNN``       : the tool return observed on the FINAL step,
  tagged to the firing tool's own ``tools.*`` leaf (the response is the tool's
  data, not the agent's). Non-final returns ride on the trajectory via their
  OPI controllable event (also under the tool), so each tool response is
  emitted exactly once, under its tool.
- ``memory_read`` / ``memory_write`` : a durable-memory read/write event.

The attack payload is NOT exposed (the target only exposes injection points;
attacks are an attacker concern), and model identity is not an observable
(it is a construction concern, read-only via the model the agent uses).
"""

from __future__ import annotations

from superred.core.types.observable import Observable
from superred.core.types.security_domain import SecurityDomainTag

from asb_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    MEMORY_TAG,
    SYSTEM_PROMPT_TAG,
    TOOLS_TAG,
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

TOOL_CATALOG_LISTING_OBS = Observable(
    name="tool_catalog_listing",
    security_domain=TOOLS_TAG,
    description=(
        "JSON snapshot of the agent's tool catalogue: each entry is "
        "{name, description}, including the attacker tool once registered."
    ),
    observable_type="json",
)

STATIC_OBSERVABLE_SPECS: tuple[Observable, ...] = (
    SYSTEM_PROMPT_OBS,
    TOOL_CATALOG_LISTING_OBS,
)

# ---------------------------------------------------------------------------
# Dynamic observable builders (one per occurrence; index in the name)
# ---------------------------------------------------------------------------


def agent_plan_observable() -> Observable:
    """The planning workflow JSON the model generated (emitted once per run)."""
    return Observable(
        name="agent_plan",
        security_domain=AGENT_TRACE_MESSAGES_TAG,
        description="The plan-of-steps workflow JSON the agent generated.",
        observable_type="json",
    )


def agent_model_output_observable(step_index: int) -> Observable:
    """One per-step model output (the agent's [Thinking] text at *step_index*)."""
    return Observable(
        name=f"agent_model_output_{step_index:04d}",
        security_domain=AGENT_TRACE_MESSAGES_TAG,
        description=f"The agent's model output at step {step_index}.",
        observable_type="text",
    )


def agent_tool_call_observable(call_index: int) -> Observable:
    """One executed tool-call decision the agent made."""
    return Observable(
        name=f"agent_tool_call_{call_index:04d}",
        security_domain=AGENT_TRACE_TOOL_CALLS_TAG,
        description=f"One tool call the agent executed (position {call_index}).",
        observable_type="json",
    )


def tool_response_observable(call_index: int, tag: SecurityDomainTag) -> Observable:
    """The tool return observed on the FINAL step, tagged to the firing tool's
    own boundary (*tag*, a ``tools.*`` leaf), since the response is the tool's
    data. Non-final returns are recorded via their OPI controllable event (also
    under the tool), so each tool response is emitted exactly once under its
    tool."""
    return Observable(
        name=f"tool_response_{call_index:04d}",
        security_domain=tag,
        description=(f"The return value the tool produced for final-step call {call_index}."),
        observable_type="text",
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
    "TOOL_CATALOG_LISTING_OBS",
    "STATIC_OBSERVABLE_SPECS",
    "agent_plan_observable",
    "agent_model_output_observable",
    "agent_tool_call_observable",
    "tool_response_observable",
    "memory_event_observable",
]
