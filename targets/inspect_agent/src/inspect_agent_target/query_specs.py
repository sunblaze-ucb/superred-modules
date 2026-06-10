"""QuerySpec declarations for :class:`InspectAgentTarget`.

Post-run string readers (per the framework's ``Target.query -> str``
contract).  A Task bound to the concrete target should read the typed
``InspectAgentTarget.messages`` property for the rich ``list[ChatMessage]``;
these string slots are for generic / optimizer-facing access.

- ``last_response``: the final assistant text (plain string, not JSON-quoted).
- ``conversation_history``: JSON-encoded list of message dicts.
- ``function_call_trace``: JSON-encoded list of {function, arguments, id}
  for every tool call the agent emitted, in order.
"""

from __future__ import annotations

from superred.core.types.state import QuerySpec

LAST_RESPONSE_SPEC: QuerySpec = QuerySpec(
    name="last_response",
    description=(
        "Final assistant text content as a plain string (NOT JSON-quoted).  "
        "Empty when the agent produced no text."
    ),
)

CONVERSATION_HISTORY_SPEC: QuerySpec = QuerySpec(
    name="conversation_history",
    description="JSON-encoded list of message dicts (role, content, tool_calls).",
)

FUNCTION_CALL_TRACE_SPEC: QuerySpec = QuerySpec(
    name="function_call_trace",
    description=(
        "JSON-encoded list of {function, arguments, id} dicts for every tool "
        "call the agent emitted, in invocation order."
    ),
)


QUERY_SPECS: list[QuerySpec] = [
    LAST_RESPONSE_SPEC,
    CONVERSATION_HISTORY_SPEC,
    FUNCTION_CALL_TRACE_SPEC,
]
"""All QuerySpecs returned from :attr:`InspectAgentTarget.query_specs`."""

QUERY_SPEC_NAMES: frozenset[str] = frozenset(s.name for s in QUERY_SPECS)
"""Names of every query slot, for O(1) validation in ``query``."""


__all__ = [
    "LAST_RESPONSE_SPEC",
    "CONVERSATION_HISTORY_SPEC",
    "FUNCTION_CALL_TRACE_SPEC",
    "QUERY_SPECS",
    "QUERY_SPEC_NAMES",
]
