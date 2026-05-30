"""QuerySpec declarations for :class:`AgentDojoTarget`.

Post-run readers used by :class:`Task.evaluate` to score a run.  All
return values are JSON-encoded strings consistent with the framework's
``Target.query -> str`` contract.

Slot summary:

- ``last_response``: assistant final text content (concatenation of the
  ``text`` content blocks of the final assistant message; ``""`` when
  the run hit max retries with no output).
- ``function_call_trace``: JSON-encoded ``list[FunctionCall]`` covering
  every call the agent attempted (suite-prefixed names retained;
  SecurityClaim Tasks strip the ``{suite}__`` prefix before passing the
  trace to upstream's ``*_from_traces`` methods).
- ``pre_environment_snapshot``: JSON-encoded
  :class:`CompositeEnvironment.model_dump` taken at run start, after
  ``configure_target`` and after Phases 1+2 (system/user prompt events)
  but before tools are invoked.
- ``post_environment_snapshot``: JSON-encoded composite env at run end.
- ``conversation_history``: JSON-encoded full ChatMessage list.
- ``tool_catalog_snapshot``: JSON-encoded post-run catalog snapshot
  (names + descriptions + parameter schemas + kind).
- ``write_calls_made``: JSON-encoded ``list[str]`` of write-side tool
  names invoked.  Cheaper for Layer-2 predicates than parsing the full
  trace.
"""

from __future__ import annotations

from superred.core.types.state import QuerySpec

LAST_RESPONSE_SPEC: QuerySpec = QuerySpec(
    name="last_response",
    description=(
        "Assistant final text content as a plain string (NOT JSON-quoted).  "
        "Empty when the agent produced no text or the run hit the 3-retry "
        "max with no output."
    ),
)

FUNCTION_CALL_TRACE_SPEC: QuerySpec = QuerySpec(
    name="function_call_trace",
    description=(
        "JSON-encoded list of FunctionCall dicts the agent emitted, in "
        "invocation order.  Function names retain the {suite}__ prefix; "
        "consumers using upstream AgentDojo *_from_traces methods must "
        "strip the prefix before passing the trace."
    ),
)

PRE_ENVIRONMENT_SNAPSHOT_SPEC: QuerySpec = QuerySpec(
    name="pre_environment_snapshot",
    description=(
        "JSON-encoded CompositeEnvironment.model_dump captured at run "
        "start after configure_target and after Phase 1+2 (system/user "
        "prompt events) but before any tool calls."
    ),
)

POST_ENVIRONMENT_SNAPSHOT_SPEC: QuerySpec = QuerySpec(
    name="post_environment_snapshot",
    description=(
        "JSON-encoded CompositeEnvironment.model_dump captured at run "
        "end (after all tool calls have been applied)."
    ),
)

CONVERSATION_HISTORY_SPEC: QuerySpec = QuerySpec(
    name="conversation_history",
    description=(
        "JSON-encoded list of ChatMessage dicts (the full message list "
        "returned by the underlying AgentDojo AgentPipeline.query)."
    ),
)

TOOL_CATALOG_SNAPSHOT_SPEC: QuerySpec = QuerySpec(
    name="tool_catalog_snapshot",
    description=(
        "JSON-encoded list of {name, description, kind, parameters_schema} "
        "entries for the post-run tool catalog (after any attacker edits)."
    ),
)

WRITE_CALLS_MADE_SPEC: QuerySpec = QuerySpec(
    name="write_calls_made",
    description=(
        "JSON-encoded list[str] of prefixed write-side tool names the "
        "agent invoked.  Empty when the agent made no mutating calls."
    ),
)


QUERY_SPECS: list[QuerySpec] = [
    LAST_RESPONSE_SPEC,
    FUNCTION_CALL_TRACE_SPEC,
    PRE_ENVIRONMENT_SNAPSHOT_SPEC,
    POST_ENVIRONMENT_SNAPSHOT_SPEC,
    CONVERSATION_HISTORY_SPEC,
    TOOL_CATALOG_SNAPSHOT_SPEC,
    WRITE_CALLS_MADE_SPEC,
]
"""All QuerySpecs returned from :attr:`AgentDojoTarget.query_specs`,
in stable order."""

QUERY_SPEC_NAMES: frozenset[str] = frozenset(s.name for s in QUERY_SPECS)
"""Names of every query slot, for O(1) validation in ``query``."""


__all__ = [
    "LAST_RESPONSE_SPEC",
    "FUNCTION_CALL_TRACE_SPEC",
    "PRE_ENVIRONMENT_SNAPSHOT_SPEC",
    "POST_ENVIRONMENT_SNAPSHOT_SPEC",
    "CONVERSATION_HISTORY_SPEC",
    "TOOL_CATALOG_SNAPSHOT_SPEC",
    "WRITE_CALLS_MADE_SPEC",
    "QUERY_SPECS",
    "QUERY_SPEC_NAMES",
]
