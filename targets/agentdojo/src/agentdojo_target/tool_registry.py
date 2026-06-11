"""Tool registry: suite-prefixed union of all AgentDojo v1 tools.

Walks each suite's :attr:`TaskSuite.tools` list and produces a flat
catalog of disambiguated :class:`agentdojo.functions_runtime.Function`
instances:

- Tools are renamed to ``{suite}__{tool}`` (e.g. ``workspace__send_email``).
- Each tool's ``Depends`` extractor is rebound to navigate the
  :class:`CompositeEnvironment` sub-attribute for its suite, so a
  workspace tool's ``Depends("inbox")`` becomes a callable
  ``lambda env: env.workspace.inbox`` against the composite root.
- Each tool is classified as read-only or mutating; the classification
  is hardcoded here and asserted exhaustive against the upstream
  ``tools`` list at module-import time.  Any new tool added upstream
  will raise ``RegistryMismatchError`` so the port catches drift
  immediately rather than silently producing a wrong injection map.

The 2024-12 upstream catalog has 74 tools: banking 11, workspace 24
(including the semi-mutating ``get_unread_emails`` which we class as
read for injection purposes), slack 11 (including ``get_webpage`` which
audit-trails its calls but never mutates content), travel 28.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Pre-import to flush the AgentDojo registration chain in the right order
# (see env.py for the explanation).
import agentdojo.task_suite.load_suites  # noqa: F401
from agentdojo.default_suites.v1.banking.task_suite import task_suite as banking_suite
from agentdojo.default_suites.v1.slack.task_suite import task_suite as slack_suite
from agentdojo.default_suites.v1.travel.task_suite import task_suite as travel_suite
from agentdojo.default_suites.v1.workspace.task_suite import task_suite as workspace_suite
from agentdojo.functions_runtime import Depends, Function
from pydantic import BaseModel

SUITE_NAMES: tuple[str, str, str, str] = ("banking", "workspace", "slack", "travel")

_NAME_SEPARATOR: str = "__"

# ---------------------------------------------------------------------------
# Read/write classification per suite.
#
# Read = the tool returns environment data the agent reads but does not
# semantically modify.  Some "read" tools have benign side effects (e.g.
# ``workspace.get_unread_emails`` flips the read flag,
# ``slack.get_webpage`` appends to the request log).  Those side effects
# are preserved verbatim; the tool is still classed as a read for the
# purpose of injection, and the side effect surfaces as observable env
# diff at run end.
#
# Write = the tool semantically mutates state on the user's behalf
# (sending money, sending emails, creating files, posting to slack,
# making reservations, etc.).  Write tools never receive on-demand
# injection: the agent's invocation is allowed to execute its
# legitimate effect, and the call lands on the trajectory via the
# live per-call ``agent_trace_tool_call`` observables (plus the
# ``write_calls_made`` query for evaluators).
#
# These sets must enumerate *exactly* the upstream tools per suite;
# build_registry() asserts both directions (no missing, no extra).
# ---------------------------------------------------------------------------

READ_TOOLS: dict[str, frozenset[str]] = {
    "banking": frozenset({
        "get_iban",
        "get_balance",
        "get_most_recent_transactions",
        "get_scheduled_transactions",
        "read_file",
        "get_user_info",
    }),
    "workspace": frozenset({
        # email reads
        "search_emails",
        "get_sent_emails",
        "get_received_emails",
        "get_draft_emails",
        "search_contacts_by_name",
        "search_contacts_by_email",
        "get_unread_emails",  # semi-mutating: flips read flag; still injectable
        # calendar reads
        "get_day_calendar_events",
        "search_calendar_events",
        "get_current_day",
        # cloud-drive reads
        "search_files_by_filename",
        "get_file_by_id",
        "list_files",
        "search_files",
    }),
    "slack": frozenset({
        "get_channels",
        "read_channel_messages",
        "read_inbox",
        "get_users_in_channel",
        "get_webpage",  # audit-trail write to web_requests; content read
    }),
    "travel": frozenset({
        "get_user_information",
        # hotels
        "get_all_hotels_in_city",
        "get_hotels_prices",
        "get_hotels_address",
        "get_rating_reviews_for_hotels",
        # restaurants
        "get_all_restaurants_in_city",
        "get_restaurants_address",
        "get_rating_reviews_for_restaurants",
        "get_cuisine_type_for_restaurants",
        "get_dietary_restrictions_for_all_restaurants",
        "get_contact_information_for_restaurants",
        "get_price_for_restaurants",
        "check_restaurant_opening_hours",
        # car rentals
        "get_all_car_rental_companies_in_city",
        "get_car_types_available",
        "get_rating_reviews_for_car_rental",
        "get_car_rental_address",
        "get_car_fuel_options",
        "get_car_price_per_day",
        # flights
        "get_flight_information",
        # calendar
        "get_day_calendar_events",
        "search_calendar_events",
    }),
}

WRITE_TOOLS: dict[str, frozenset[str]] = {
    "banking": frozenset({
        "send_money",
        "schedule_transaction",
        "update_scheduled_transaction",
        "update_password",
        "update_user_info",
    }),
    "workspace": frozenset({
        "send_email",
        "delete_email",
        "create_calendar_event",
        "cancel_calendar_event",
        "reschedule_calendar_event",
        "add_calendar_event_participants",
        "create_file",
        "delete_file",
        "share_file",
        "append_to_file",
    }),
    "slack": frozenset({
        "add_user_to_channel",
        "send_direct_message",
        "send_channel_message",
        "invite_user_to_slack",
        "remove_user_from_slack",
        "post_webpage",
    }),
    "travel": frozenset({
        "reserve_hotel",
        "reserve_restaurant",
        "reserve_car_rental",
        "create_calendar_event",
        "cancel_calendar_event",
        "send_email",
    }),
}


class RegistryMismatchError(RuntimeError):
    """Raised at import time when the hardcoded R/W classification
    diverges from the upstream tool list.

    Two failure modes:
    - Tool in the upstream suite that the registry does not classify
      (the classification table needs an entry added).
    - Tool in the registry table but not in upstream (the upstream
      removed/renamed something and the table is stale).
    """


# ---------------------------------------------------------------------------
# Suite handles
# ---------------------------------------------------------------------------

_SUITES_BY_NAME: dict[str, object] = {
    "banking": banking_suite,
    "workspace": workspace_suite,
    "slack": slack_suite,
    "travel": travel_suite,
}


# ---------------------------------------------------------------------------
# Name prefixing helpers
# ---------------------------------------------------------------------------


def prefixed_name(suite: str, tool: str) -> str:
    """Return the disambiguated catalog name ``{suite}__{tool}``."""
    return f"{suite}{_NAME_SEPARATOR}{tool}"


def split_prefixed(name: str) -> tuple[str, str]:
    """Inverse of :func:`prefixed_name`: split ``suite__tool`` into ``(suite, tool)``.

    Raises:
        ValueError: If *name* is not in the prefixed form (no separator
            or unknown suite).
    """
    if _NAME_SEPARATOR not in name:
        raise ValueError(f"Name {name!r} is not in 'suite__tool' form")
    suite, _, tool = name.partition(_NAME_SEPARATOR)
    if suite not in SUITE_NAMES:
        raise ValueError(f"Unknown suite prefix in {name!r}: {suite!r}")
    return suite, tool


# ---------------------------------------------------------------------------
# Dependency rebinding
# ---------------------------------------------------------------------------


def _rebind_string_dep(suite: str, attr_name: str) -> Callable[[BaseModel], BaseModel]:
    """Build an extractor that returns ``env.<suite>.<attr_name>``.

    Defined as a named factory (not an inline lambda inside a loop)
    so the closure captures ``suite`` and ``attr_name`` by value rather
    than by reference.
    """
    def extract(env: BaseModel) -> BaseModel:
        sub = getattr(env, suite)
        return getattr(sub, attr_name)
    return extract


def _rebind_callable_dep(
    suite: str, original: Callable[[BaseModel], BaseModel]
) -> Callable[[BaseModel], BaseModel]:
    """Wrap a callable Depends extractor so it receives the suite sub-env.

    Upstream AgentDojo only uses string deps in v1, but the framework
    permits callables.  We handle both for forward compatibility.
    """
    def extract(env: BaseModel) -> BaseModel:
        return original(getattr(env, suite))
    return extract


def _rebind_dependencies(
    suite: str, deps: dict[str, Depends]
) -> dict[str, Depends]:
    """Return a new ``{arg_name: Depends}`` dict with each Depends rebound
    to navigate ``env.<suite>.<original>``."""
    rebound: dict[str, Depends] = {}
    for arg_name, dep in deps.items():
        env_dependency = dep.env_dependency
        if isinstance(env_dependency, str):
            extractor = _rebind_string_dep(suite, env_dependency)
        elif callable(env_dependency):
            extractor = _rebind_callable_dep(suite, env_dependency)
        else:
            raise TypeError(
                f"Unsupported Depends.env_dependency type for "
                f"{suite}.{arg_name}: {type(env_dependency).__name__}"
            )
        rebound[arg_name] = Depends(env_dependency=extractor)
    return rebound


# ---------------------------------------------------------------------------
# Function rebuilding
# ---------------------------------------------------------------------------


def _rebuild_function(suite: str, original: Function) -> Function:
    """Construct a new :class:`Function` whose name is suite-prefixed and
    whose dependencies are rebound for the composite environment.

    All other fields (``description``, ``parameters``, ``run``,
    ``full_docstring``, ``return_type``) are reused unchanged.
    """
    return Function(
        name=prefixed_name(suite, original.name),
        description=original.description,
        parameters=original.parameters,
        dependencies=_rebind_dependencies(suite, original.dependencies),
        run=original.run,
        full_docstring=original.full_docstring,
        return_type=original.return_type,
    )


# ---------------------------------------------------------------------------
# Classification assertions
# ---------------------------------------------------------------------------


def _classify(suite: str, tool_name: str) -> str:
    """Return ``"read"`` or ``"write"`` for the named tool in the named suite.

    Raises:
        RegistryMismatchError: If the tool is classified in neither set.
    """
    reads = READ_TOOLS[suite]
    writes = WRITE_TOOLS[suite]
    in_reads = tool_name in reads
    in_writes = tool_name in writes
    if in_reads and in_writes:
        raise RegistryMismatchError(
            f"Tool {tool_name!r} in suite {suite!r} appears in both READ_TOOLS "
            "and WRITE_TOOLS classifications.  Each tool must belong to exactly "
            "one set; fix the table."
        )
    if not (in_reads or in_writes):
        raise RegistryMismatchError(
            f"Tool {tool_name!r} in suite {suite!r} is registered upstream but "
            "missing from both READ_TOOLS and WRITE_TOOLS.  Add it to the "
            "appropriate set in tool_registry.py."
        )
    return "read" if in_reads else "write"


def _assert_no_stale_entries(suite: str, upstream_names: set[str]) -> None:
    """Raise if READ_TOOLS or WRITE_TOOLS lists a tool absent upstream."""
    classified = READ_TOOLS[suite] | WRITE_TOOLS[suite]
    stale = classified - upstream_names
    if stale:
        raise RegistryMismatchError(
            f"Suite {suite!r}: classification lists tools missing from upstream: "
            f"{sorted(stale)}.  Remove them from READ_TOOLS / WRITE_TOOLS."
        )


# ---------------------------------------------------------------------------
# Public registry record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryEntry:
    """One entry in :data:`TOOL_REGISTRY`.

    Attributes:
        prefixed_name: The disambiguated catalog name (``{suite}__{tool}``).
        suite: One of ``banking``, ``workspace``, ``slack``, ``travel``.
        original_name: The upstream tool name without the suite prefix.
        kind: ``"read"`` or ``"write"``.
        function: The rebuilt :class:`Function` ready to register with
            a :class:`FunctionsRuntime`.
    """

    prefixed_name: str
    suite: str
    original_name: str
    kind: str
    function: Function


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, RegistryEntry]:
    """Build the full prefixed-tool registry at module import time.

    Validates the R/W classification against each suite's upstream
    ``tools`` list (both directions: no missing, no extra).
    """
    entries: dict[str, RegistryEntry] = {}
    for suite_name in SUITE_NAMES:
        suite_obj = _SUITES_BY_NAME[suite_name]
        upstream_tools: list[Function] = suite_obj.tools  # type: ignore[attr-defined]
        upstream_names: set[str] = {t.name for t in upstream_tools}
        _assert_no_stale_entries(suite_name, upstream_names)
        for original in upstream_tools:
            kind = _classify(suite_name, original.name)
            new_fn = _rebuild_function(suite_name, original)
            entry = RegistryEntry(
                prefixed_name=new_fn.name,
                suite=suite_name,
                original_name=original.name,
                kind=kind,
                function=new_fn,
            )
            if entry.prefixed_name in entries:
                raise RegistryMismatchError(
                    f"Duplicate prefixed name: {entry.prefixed_name!r}"
                )
            entries[entry.prefixed_name] = entry
    return entries


TOOL_REGISTRY: dict[str, RegistryEntry] = _build_registry()
"""Flat map of ``{suite}__{tool}`` -> :class:`RegistryEntry` covering every
upstream v1 tool.  Module-level constant: built once at import time."""

ALL_FUNCTIONS: list[Function] = [e.function for e in TOOL_REGISTRY.values()]
"""Convenience list of every rebound :class:`Function` in registration order."""

READ_FUNCTION_NAMES: frozenset[str] = frozenset(
    e.prefixed_name for e in TOOL_REGISTRY.values() if e.kind == "read"
)
"""Prefixed names of every read tool.  Used by the runtime wrapper to
decide whether to fire a per-read :class:`ControllablePostCallEvent`."""

WRITE_FUNCTION_NAMES: frozenset[str] = frozenset(
    e.prefixed_name for e in TOOL_REGISTRY.values() if e.kind == "write"
)
"""Prefixed names of every write tool.  Used by the target's
``write_calls_made`` query to report mutating calls to evaluators."""


__all__ = [
    "SUITE_NAMES",
    "READ_TOOLS",
    "WRITE_TOOLS",
    "RegistryEntry",
    "RegistryMismatchError",
    "TOOL_REGISTRY",
    "ALL_FUNCTIONS",
    "READ_FUNCTION_NAMES",
    "WRITE_FUNCTION_NAMES",
    "prefixed_name",
    "split_prefixed",
]
