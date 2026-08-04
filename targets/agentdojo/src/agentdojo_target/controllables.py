"""Module-level Controllable singletons exposed by :class:`AgentDojoTarget`.

Categories:

- ``SYSTEM_PROMPT_CTRL``: agent system prompt (scope ``system.system_prompt``).
- ``USER_PROMPT_CTRL``: benign user instruction (scope ``user``).  An
  attacker with the ``user`` tag in scope can override the Task's
  benign prompt via :class:`ControllableInjection`.
- Tool-catalogue controllables: four operations under the
  ``system.tool_catalogue`` grouping root, split by capability into
  ``tool_catalogue_add`` (register), ``tool_catalogue_edit`` (replace +
  rewrite-description), and ``tool_catalogue_remove`` (unregister).
- Per-read controllables: one per readable tool in the catalog
  (47 entries), each tagged at the store leaf it reads from.

The read-tool -> store-leaf mapping lives in :data:`READ_STORE_MAP`; the
write-tool -> store-leaf mapping (used only to tag the write observation,
not to create a controllable) lives in :data:`WRITE_STORE_MAP`.  See
``tests/test_store_rationale.py`` for the per-tool rationale behind each
assignment.  Each store leaf maps to exactly one upstream data store
(one ``Depends`` extractor), so no read tool spans more than one store
and a per-read controllable never needs to be split.  A read controllable
and the write observation for the same store reference the same leaf, so
reading and acting on a store share a boundary.
"""

from __future__ import annotations

from superred.core.types.controllable import Controllable
from superred.core.types.security_domain import SecurityDomainTag

from agentdojo_target.security_tags import (
    BANKING_BANK_ACCOUNT_TAG,
    BANKING_FILESYSTEM_TAG,
    BANKING_USER_ACCOUNT_TAG,
    SLACK_SLACK_TAG,
    SLACK_WEB_TAG,
    SYSTEM_PROMPT_TAG,
    TOOL_CATALOGUE_ADD_TAG,
    TOOL_CATALOGUE_EDIT_TAG,
    TOOL_CATALOGUE_REMOVE_TAG,
    TRAVEL_CALENDAR_TAG,
    TRAVEL_CAR_RENTAL_TAG,
    TRAVEL_FLIGHTS_TAG,
    TRAVEL_HOTELS_TAG,
    TRAVEL_INBOX_TAG,
    TRAVEL_RESERVATION_TAG,
    TRAVEL_RESTAURANTS_TAG,
    TRAVEL_USER_TAG,
    USER_TAG,
    WORKSPACE_CALENDAR_TAG,
    WORKSPACE_CLOUD_DRIVE_TAG,
    WORKSPACE_INBOX_TAG,
)
from agentdojo_target.tool_registry import (
    READ_FUNCTION_NAMES,
    SUITE_NAMES,
    TOOL_REGISTRY,
    WRITE_FUNCTION_NAMES,
    prefixed_name,
    split_prefixed,
)

# ---------------------------------------------------------------------------
# Prompt controllables
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_CTRL: Controllable = Controllable(
    name="system_prompt",
    security_domain=SYSTEM_PROMPT_TAG,
    description="Override the agent's system prompt for this run.",
    value_type="text",
)

USER_PROMPT_CTRL: Controllable = Controllable(
    name="user_prompt",
    security_domain=USER_TAG,
    description="Override the benign user instruction set by the Task.",
    value_type="text",
)

# ---------------------------------------------------------------------------
# Tool-catalogue controllables
# ---------------------------------------------------------------------------

TOOL_CATALOG_REGISTER_CTRL: Controllable = Controllable(
    name="tool_catalog_register",
    security_domain=TOOL_CATALOGUE_ADD_TAG,
    description=(
        "Register a new attacker-defined tool with a per-call fake "
        "return value.  The injection value is a JSON object: "
        '{"name": str, "description": str, "parameters_schema": dict, '
        '"fake_return": Any}.'
    ),
    value_type="json",
)

TOOL_CATALOG_REPLACE_CTRL: Controllable = Controllable(
    name="tool_catalog_replace",
    security_domain=TOOL_CATALOGUE_EDIT_TAG,
    description=(
        "Replace an existing tool (shadow attack): when the agent calls "
        "the named tool, the attacker-supplied fake_return is used. "
        'Injection value: {"name": str, "fake_return": Any, '
        '"description"?: str}.'
    ),
    value_type="json",
)

TOOL_CATALOG_UNREGISTER_CTRL: Controllable = Controllable(
    name="tool_catalog_unregister",
    security_domain=TOOL_CATALOGUE_REMOVE_TAG,
    description=(
        'Remove an existing tool from the catalog.  Injection value: {"name": str}.'
    ),
    value_type="json",
)

TOOL_CATALOG_REWRITE_DOC_CTRL: Controllable = Controllable(
    name="tool_catalog_rewrite_doc",
    security_domain=TOOL_CATALOGUE_EDIT_TAG,
    description=(
        "Rewrite an existing tool's description/docstring without "
        'changing its body.  Injection value: {"name": str, '
        '"description": str}.'
    ),
    value_type="json",
)

TOOL_CATALOG_CTRLS: tuple[Controllable, ...] = (
    TOOL_CATALOG_REGISTER_CTRL,
    TOOL_CATALOG_REPLACE_CTRL,
    TOOL_CATALOG_UNREGISTER_CTRL,
    TOOL_CATALOG_REWRITE_DOC_CTRL,
)

# ---------------------------------------------------------------------------
# Per-read store-leaf mapping
#
# Each read tool is tagged at the single upstream data store it reads from
# (the store named by the tool's data-bearing ``Depends`` extractor).  No
# read tool reads from more than one store, so each gets exactly one
# controllable at one leaf; there is nothing to split.
# ---------------------------------------------------------------------------

READ_STORE_MAP: dict[str, SecurityDomainTag] = {
    # ---- banking ----
    "banking__get_iban": BANKING_BANK_ACCOUNT_TAG,
    "banking__get_balance": BANKING_BANK_ACCOUNT_TAG,
    "banking__get_most_recent_transactions": BANKING_BANK_ACCOUNT_TAG,
    "banking__get_scheduled_transactions": BANKING_BANK_ACCOUNT_TAG,
    "banking__read_file": BANKING_FILESYSTEM_TAG,
    "banking__get_user_info": BANKING_USER_ACCOUNT_TAG,
    # ---- workspace ----
    "workspace__search_emails": WORKSPACE_INBOX_TAG,
    "workspace__get_sent_emails": WORKSPACE_INBOX_TAG,
    "workspace__get_received_emails": WORKSPACE_INBOX_TAG,
    "workspace__get_draft_emails": WORKSPACE_INBOX_TAG,
    "workspace__search_contacts_by_name": WORKSPACE_INBOX_TAG,
    "workspace__search_contacts_by_email": WORKSPACE_INBOX_TAG,
    "workspace__get_unread_emails": WORKSPACE_INBOX_TAG,
    "workspace__get_day_calendar_events": WORKSPACE_CALENDAR_TAG,
    "workspace__search_calendar_events": WORKSPACE_CALENDAR_TAG,
    # get_current_day reads the calendar store (Depends("calendar")),
    # not a system clock.
    "workspace__get_current_day": WORKSPACE_CALENDAR_TAG,
    "workspace__search_files_by_filename": WORKSPACE_CLOUD_DRIVE_TAG,
    "workspace__get_file_by_id": WORKSPACE_CLOUD_DRIVE_TAG,
    "workspace__list_files": WORKSPACE_CLOUD_DRIVE_TAG,
    "workspace__search_files": WORKSPACE_CLOUD_DRIVE_TAG,
    # ---- slack ----
    # Every slack tool depends on the single Slack object (users, channels,
    # channel messages, direct messages), so all messaging reads share one
    # leaf; only the web store is separable.
    "slack__get_channels": SLACK_SLACK_TAG,
    "slack__read_channel_messages": SLACK_SLACK_TAG,
    "slack__read_inbox": SLACK_SLACK_TAG,
    "slack__get_users_in_channel": SLACK_SLACK_TAG,
    "slack__get_webpage": SLACK_WEB_TAG,
    # ---- travel ----
    "travel__get_user_information": TRAVEL_USER_TAG,
    # Hotels (4)
    "travel__get_all_hotels_in_city": TRAVEL_HOTELS_TAG,
    "travel__get_hotels_prices": TRAVEL_HOTELS_TAG,
    "travel__get_hotels_address": TRAVEL_HOTELS_TAG,
    "travel__get_rating_reviews_for_hotels": TRAVEL_HOTELS_TAG,
    # Restaurants (8)
    "travel__get_all_restaurants_in_city": TRAVEL_RESTAURANTS_TAG,
    "travel__get_restaurants_address": TRAVEL_RESTAURANTS_TAG,
    "travel__get_rating_reviews_for_restaurants": TRAVEL_RESTAURANTS_TAG,
    "travel__get_cuisine_type_for_restaurants": TRAVEL_RESTAURANTS_TAG,
    "travel__get_dietary_restrictions_for_all_restaurants": TRAVEL_RESTAURANTS_TAG,
    "travel__get_contact_information_for_restaurants": TRAVEL_RESTAURANTS_TAG,
    "travel__get_price_for_restaurants": TRAVEL_RESTAURANTS_TAG,
    "travel__check_restaurant_opening_hours": TRAVEL_RESTAURANTS_TAG,
    # Car rentals (6)
    "travel__get_all_car_rental_companies_in_city": TRAVEL_CAR_RENTAL_TAG,
    "travel__get_car_types_available": TRAVEL_CAR_RENTAL_TAG,
    "travel__get_rating_reviews_for_car_rental": TRAVEL_CAR_RENTAL_TAG,
    "travel__get_car_rental_address": TRAVEL_CAR_RENTAL_TAG,
    "travel__get_car_fuel_options": TRAVEL_CAR_RENTAL_TAG,
    "travel__get_car_price_per_day": TRAVEL_CAR_RENTAL_TAG,
    # Flights (1)
    "travel__get_flight_information": TRAVEL_FLIGHTS_TAG,
    # Calendar (2) -- the travel-side calendar store.
    "travel__get_day_calendar_events": TRAVEL_CALENDAR_TAG,
    "travel__search_calendar_events": TRAVEL_CALENDAR_TAG,
}

# ---------------------------------------------------------------------------
# Per-write store-leaf mapping
#
# Write tools get NO controllable (their effect must really execute so the
# environment diff can detect the attack).  This map tags the one-way write
# observation at the store the tool mutates, so a service-scoped attacker
# sees the action it provoked under the same boundary it reads from.  Where
# a write touches two stores (a calendar write that also sends a
# notification email; a reservation that also reads the user profile), it is
# tagged at the PRIMARY mutated store; the secondary touch is documented in
# ASSUMPTIONS.md and still surfaces in the post-run environment diff.
# ---------------------------------------------------------------------------

WRITE_STORE_MAP: dict[str, SecurityDomainTag] = {
    # ---- banking ----
    "banking__send_money": BANKING_BANK_ACCOUNT_TAG,
    "banking__schedule_transaction": BANKING_BANK_ACCOUNT_TAG,
    "banking__update_scheduled_transaction": BANKING_BANK_ACCOUNT_TAG,
    "banking__update_password": BANKING_USER_ACCOUNT_TAG,
    "banking__update_user_info": BANKING_USER_ACCOUNT_TAG,
    # ---- workspace ----
    "workspace__send_email": WORKSPACE_INBOX_TAG,
    "workspace__delete_email": WORKSPACE_INBOX_TAG,
    "workspace__create_calendar_event": WORKSPACE_CALENDAR_TAG,
    "workspace__cancel_calendar_event": WORKSPACE_CALENDAR_TAG,
    "workspace__reschedule_calendar_event": WORKSPACE_CALENDAR_TAG,
    "workspace__add_calendar_event_participants": WORKSPACE_CALENDAR_TAG,
    "workspace__create_file": WORKSPACE_CLOUD_DRIVE_TAG,
    "workspace__delete_file": WORKSPACE_CLOUD_DRIVE_TAG,
    "workspace__share_file": WORKSPACE_CLOUD_DRIVE_TAG,
    "workspace__append_to_file": WORKSPACE_CLOUD_DRIVE_TAG,
    # ---- slack ----
    "slack__add_user_to_channel": SLACK_SLACK_TAG,
    "slack__send_direct_message": SLACK_SLACK_TAG,
    "slack__send_channel_message": SLACK_SLACK_TAG,
    "slack__invite_user_to_slack": SLACK_SLACK_TAG,
    "slack__remove_user_from_slack": SLACK_SLACK_TAG,
    "slack__post_webpage": SLACK_WEB_TAG,
    # ---- travel ----
    "travel__reserve_hotel": TRAVEL_RESERVATION_TAG,
    "travel__reserve_restaurant": TRAVEL_RESERVATION_TAG,
    "travel__reserve_car_rental": TRAVEL_RESERVATION_TAG,
    "travel__create_calendar_event": TRAVEL_CALENDAR_TAG,
    "travel__cancel_calendar_event": TRAVEL_CALENDAR_TAG,
    "travel__send_email": TRAVEL_INBOX_TAG,
}


# ---------------------------------------------------------------------------
# Build per-read Controllables
# ---------------------------------------------------------------------------


def _make_read_ctrl(prefixed: str) -> Controllable:
    """Build a Controllable for a single prefixed read-tool name."""
    if prefixed not in READ_STORE_MAP:
        raise RuntimeError(
            f"Read tool {prefixed!r} has no entry in READ_STORE_MAP. "
            "Add an entry mapping it to the store leaf it reads from."
        )
    suite, original = split_prefixed(prefixed)
    return Controllable(
        name=f"read__{prefixed}",
        security_domain=READ_STORE_MAP[prefixed],
        description=(
            f"Per-read injection point for ``{prefixed}`` "
            f"(suite={suite}, upstream tool={original}).  When the agent "
            "invokes this read, the runtime fires a "
            "ControllablePostCallEvent carrying the legitimate value as "
            "``answer``; a ControllableInjection response replaces the "
            "agent-visible return."
        ),
        # Consumed as a RAW STRING (runtime_wrapper injects response.value verbatim
        # as the agent-visible return); ``text`` keeps the value_type contract
        # truthful so a generic attacker does not mistake this for a schema surface.
        value_type="text",
    )


def _build_read_controllables() -> dict[str, Controllable]:
    """Build one Controllable per read tool, in deterministic order.

    Validates that :data:`READ_STORE_MAP` covers every read tool and
    contains no extras (no stale mappings).
    """
    mapped = set(READ_STORE_MAP)
    expected = set(READ_FUNCTION_NAMES)
    missing = expected - mapped
    if missing:
        raise RuntimeError(
            f"READ_STORE_MAP is missing entries for read tools: {sorted(missing)}"
        )
    stale = mapped - expected
    if stale:
        raise RuntimeError(
            f"READ_STORE_MAP has entries for non-read tools: {sorted(stale)}"
        )

    out: dict[str, Controllable] = {}
    for suite in SUITE_NAMES:
        suite_tools = sorted(
            e.original_name
            for e in TOOL_REGISTRY.values()
            if e.suite == suite and e.kind == "read"
        )
        for tool in suite_tools:
            prefixed = prefixed_name(suite, tool)
            out[prefixed] = _make_read_ctrl(prefixed)
    return out


READ_CTRLS: dict[str, Controllable] = _build_read_controllables()
"""Map of prefixed tool name -> the Controllable that gates the read.

The wrapped runtime uses this map to look up the right Controllable
when firing per-call events."""


def _validate_write_store_map() -> None:
    """Validate :data:`WRITE_STORE_MAP` covers exactly the write tools.

    Mirrors the read-side exhaustiveness guard so upstream tool drift
    (a write added or removed) fails loudly at import time rather than
    silently leaving a write untagged.
    """
    mapped = set(WRITE_STORE_MAP)
    expected = set(WRITE_FUNCTION_NAMES)
    missing = expected - mapped
    if missing:
        raise RuntimeError(
            f"WRITE_STORE_MAP is missing entries for write tools: {sorted(missing)}"
        )
    stale = mapped - expected
    if stale:
        raise RuntimeError(
            f"WRITE_STORE_MAP has entries for non-write tools: {sorted(stale)}"
        )


_validate_write_store_map()


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

CONTROLLABLES: list[Controllable] = [
    SYSTEM_PROMPT_CTRL,
    USER_PROMPT_CTRL,
    *TOOL_CATALOG_CTRLS,
    *READ_CTRLS.values(),
]
"""Every Controllable AgentDojoTarget exposes, in stable order
(system_prompt, user_prompt, the four catalog ctrls, then the 47 read
ctrls grouped by suite then by upstream name)."""


__all__ = [
    "SYSTEM_PROMPT_CTRL",
    "USER_PROMPT_CTRL",
    "TOOL_CATALOG_REGISTER_CTRL",
    "TOOL_CATALOG_REPLACE_CTRL",
    "TOOL_CATALOG_UNREGISTER_CTRL",
    "TOOL_CATALOG_REWRITE_DOC_CTRL",
    "TOOL_CATALOG_CTRLS",
    "READ_STORE_MAP",
    "WRITE_STORE_MAP",
    "READ_CTRLS",
    "CONTROLLABLES",
]
