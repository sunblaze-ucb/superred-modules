"""Observables exposed by the DTAP agent targets (shared by both agents).

Static observables are returned from ``get_observables`` (read once at init);
dynamic observables are emitted live on the trajectory during ``run`` via
``emit(ObservableEvent(observable=..., content=...))``.

Emit-once discipline: an MCP env tool's call+return is emitted exactly once, on
its server's ``env_tool:<server>`` ``ControllablePostCallEvent`` (the specific
tool + params ride in the event's ``request``; see ``controllables``), NOT
mirrored here. Only the agent's NATIVE tool calls and its non-tool messages come
from the in-container transcript, each emitted once. The trajectory extractor
SKIPS proxied tool entries so they are not double-counted.
"""

from __future__ import annotations

from superred.core.types.observable import Observable

from dtap_scaffold.forest import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    DETAILED_SYSTEM_SPECIFICATION_TAG,
    MAX_TURNS_TAG,
    MODEL_IDENTITY_TAG,
    TOOL_CATALOGUE_TAG,
)

# ---------------------------------------------------------------------------
# Static observables (available pre-run, returned from get_observables)
# ---------------------------------------------------------------------------

MODEL_IDENTITY_OBS: Observable = Observable(
    name="model_identity",
    security_domain=MODEL_IDENTITY_TAG,
    description="The model id powering the agent (construction concern, fixed per experiment).",
    observable_type="text",
)

DETAILED_SYSTEM_SPECIFICATION_OBS: Observable = Observable(
    name="detailed_system_specification",
    security_domain=DETAILED_SYSTEM_SPECIFICATION_TAG,
    description=(
        "A free-text design brief of this DTAP target: the agent runtime, the "
        "active environments, and how the run is wired. A leaked-documentation "
        "intelligence surface (read-only)."
    ),
    observable_type="text",
)

TOOL_CATALOGUE_OBS: Observable = Observable(
    name="tool_catalogue",
    security_domain=TOOL_CATALOGUE_TAG,
    description=(
        "The full per-tool catalogue of every active env tool: "
        '``{"servers": {server: [{"name", "description", "inputSchema"}]}}`` with the '
        "genuine (unedited) backend descriptions. Emitted ONCE at the start of run() "
        "(after the env boots) so the optimizer can read the whole tool surface before "
        "choosing tool-description injections. Scoped to the tool-catalogue boundary."
    ),
    observable_type="json",
)

ACTIVE_ENVIRONMENTS_OBS: Observable = Observable(
    name="active_environments",
    security_domain=DETAILED_SYSTEM_SPECIFICATION_TAG,
    description="JSON list of the environments/MCP servers active for this task.",
    observable_type="json",
)

MAX_TURNS_OBS: Observable = Observable(
    name="max_turns",
    security_domain=MAX_TURNS_TAG,
    description="The agent loop's max-turns bound for this run.",
    observable_type="text",
)

STATIC_OBSERVABLE_SPECS: tuple[Observable, ...] = (
    MODEL_IDENTITY_OBS,
    DETAILED_SYSTEM_SPECIFICATION_OBS,
    ACTIVE_ENVIRONMENTS_OBS,
    MAX_TURNS_OBS,
)
"""The static observables every DTAP target exposes (content filled per task).

The active-server list has a single home here -- ``active_environments`` (under
``detailed_system_specification`` -- knowing which envs are active is system-spec
intel). It is deliberately NOT surfaced under the tool-catalogue boundary: the
tool-catalogue tag is served by the live ``tool_catalogue`` observable (the actual
per-tool catalogue), so a tool-catalogue-scoped optimizer does not learn the
server list, which belongs to the system-spec boundary."""


# ---------------------------------------------------------------------------
# Dynamic observables (emitted live during run)
# ---------------------------------------------------------------------------


def agent_message_observable(index: int) -> Observable:
    """Observable for one NON-TOOL agent message (system/assistant/user/reasoning)
    at position *index*, reconstructed from the in-container transcript."""
    return Observable(
        name=f"agent_trace_message_{index:04d}",
        security_domain=AGENT_TRACE_MESSAGES_TAG,
        description=f"One non-tool message from the agent's run (position {index}).",
        observable_type="json",
    )


def native_tool_observable(index: int) -> Observable:
    """Observable for one NATIVE (in-container bash/edit/exec/fs) tool call+return
    at position *index*, reconstructed from the in-container transcript. MCP env
    tool calls are NOT emitted here (they fire on their PostCall controllable)."""
    return Observable(
        name=f"native_tool_call_{index:04d}",
        security_domain=AGENT_TRACE_TOOL_CALLS_TAG,
        description=(f"One native (agent-container) tool call and its return (position {index})."),
        observable_type="json",
    )


__all__ = [
    "MODEL_IDENTITY_OBS",
    "DETAILED_SYSTEM_SPECIFICATION_OBS",
    "TOOL_CATALOGUE_OBS",
    "ACTIVE_ENVIRONMENTS_OBS",
    "MAX_TURNS_OBS",
    "STATIC_OBSERVABLE_SPECS",
    "agent_message_observable",
    "native_tool_observable",
]
