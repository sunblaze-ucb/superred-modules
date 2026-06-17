"""Per-tool store rationale audit.

For every entry in :data:`agentdojo_target.controllables.READ_STORE_MAP`,
this file pins:

- the expected store leaf the tool reads from (one leaf == one upstream
  pydantic store == one ``Depends`` extractor)
- a one-line rationale grounded in the tool's data-bearing ``Depends``
  store

The parametrized test asserts that the live ``READ_STORE_MAP`` exactly
matches this expected table.  A divergence either flags an upstream
change (a new read tool) or a deliberate store reassignment that the
author must mirror here.

This serves three purposes:

1. **Verification of the store mapping**: every read tool's store
   assignment is encoded here with its justification.
2. **Living documentation**: ``STORE_RATIONALES`` is the canonical
   one-line justification per tool.  Future readers see WHICH store a
   tool reads from without having to chase the upstream ``Depends``.
3. **Drift detection**: if a new tool is added to ``READ_FUNCTION_NAMES``
   without a rationale entry here, the test fails with the missing
   tool name.
"""

from __future__ import annotations

import pytest

from agentdojo_target.controllables import READ_STORE_MAP
from agentdojo_target.security_tags import (
    BANKING_BANK_ACCOUNT_TAG,
    BANKING_FILESYSTEM_TAG,
    BANKING_USER_ACCOUNT_TAG,
    SLACK_SLACK_TAG,
    SLACK_WEB_TAG,
    TRAVEL_CALENDAR_TAG,
    TRAVEL_CAR_RENTAL_TAG,
    TRAVEL_FLIGHTS_TAG,
    TRAVEL_HOTELS_TAG,
    TRAVEL_RESTAURANTS_TAG,
    TRAVEL_USER_TAG,
    WORKSPACE_CALENDAR_TAG,
    WORKSPACE_CLOUD_DRIVE_TAG,
    WORKSPACE_INBOX_TAG,
)
from agentdojo_target.tool_registry import READ_FUNCTION_NAMES

# ---------------------------------------------------------------------------
# Per-tool (store_leaf_name, rationale) table.
#
# Store-leaf names are the .name strings of the store-leaf SecurityDomainTag
# instances under their service node (under TOOLS_TAG).  Each read tool reads
# from exactly one upstream store named by its data-bearing Depends extractor.
# ---------------------------------------------------------------------------

STORE_RATIONALES: dict[str, tuple[str, str]] = {
    # ---- banking (6) ----
    "banking__get_iban": (
        "banking_bank_account",
        "Reads the bank-account store (Depends('account')) for the IBAN.",
    ),
    "banking__get_balance": (
        "banking_bank_account",
        "Reads the bank-account store (Depends('account')) for the balance.",
    ),
    "banking__get_most_recent_transactions": (
        "banking_bank_account",
        "Reads the bank-account store's transaction history.",
    ),
    "banking__get_scheduled_transactions": (
        "banking_bank_account",
        "Reads the bank-account store's scheduled standing orders.",
    ),
    "banking__read_file": (
        "banking_filesystem",
        "Reads the banking-side filesystem store (Depends('filesystem')).",
    ),
    "banking__get_user_info": (
        "banking_user_account",
        "Reads the user-account store (Depends('user_account')) for the profile.",
    ),
    # ---- workspace (14) ----
    "workspace__search_emails": (
        "workspace_inbox",
        "Reads the inbox store (Depends('inbox')) for matching emails.",
    ),
    "workspace__get_sent_emails": (
        "workspace_inbox",
        "Reads sent items from the inbox store.",
    ),
    "workspace__get_received_emails": (
        "workspace_inbox",
        "Reads received items from the inbox store.",
    ),
    "workspace__get_draft_emails": (
        "workspace_inbox",
        "Reads drafts from the inbox store.",
    ),
    "workspace__search_contacts_by_name": (
        "workspace_inbox",
        "Reads the contact list held in the inbox store.",
    ),
    "workspace__search_contacts_by_email": (
        "workspace_inbox",
        "Reads the contact list held in the inbox store.",
    ),
    "workspace__get_unread_emails": (
        "workspace_inbox",
        "Reads unread items from the inbox store (also flips the read flag; "
        "see ASSUMPTIONS F.1).",
    ),
    "workspace__get_day_calendar_events": (
        "workspace_calendar",
        "Reads the calendar store (Depends('calendar')) for a day's events.",
    ),
    "workspace__search_calendar_events": (
        "workspace_calendar",
        "Reads the calendar store for matching events.",
    ),
    "workspace__get_current_day": (
        "workspace_calendar",
        "Reads the calendar store (Depends('calendar')) for its current_day; "
        "this is the calendar store, not a system clock.",
    ),
    "workspace__search_files_by_filename": (
        "workspace_cloud_drive",
        "Reads the cloud-drive store (Depends('cloud_drive')) by filename.",
    ),
    "workspace__get_file_by_id": (
        "workspace_cloud_drive",
        "Reads a file body from the cloud-drive store.",
    ),
    "workspace__list_files": (
        "workspace_cloud_drive",
        "Lists files in the cloud-drive store.",
    ),
    "workspace__search_files": (
        "workspace_cloud_drive",
        "Searches file content in the cloud-drive store.",
    ),
    # ---- slack (5) ----
    "slack__get_channels": (
        "slack_slack",
        "Reads channels from the single Slack store (Depends('slack')).",
    ),
    "slack__read_channel_messages": (
        "slack_slack",
        "Reads channel messages from the Slack store.",
    ),
    "slack__read_inbox": (
        "slack_slack",
        "Reads direct messages from the Slack store.",
    ),
    "slack__get_users_in_channel": (
        "slack_slack",
        "Reads the user list from the Slack store.",
    ),
    "slack__get_webpage": (
        "slack_web",
        "Reads the web store (Depends('web')) for fetched page content.",
    ),
    # ---- travel (22) ----
    "travel__get_user_information": (
        "travel_user",
        "Reads the traveller profile store (Depends('user')) for PII.",
    ),
    # Hotels (4)
    "travel__get_all_hotels_in_city": (
        "travel_hotels",
        "Reads the hotels store (Depends('hotels')) for city listings.",
    ),
    "travel__get_hotels_prices": (
        "travel_hotels",
        "Reads hotel prices from the hotels store.",
    ),
    "travel__get_hotels_address": (
        "travel_hotels",
        "Reads hotel addresses from the hotels store.",
    ),
    "travel__get_rating_reviews_for_hotels": (
        "travel_hotels",
        "Reads hotel ratings/reviews from the hotels store.",
    ),
    # Restaurants (8)
    "travel__get_all_restaurants_in_city": (
        "travel_restaurants",
        "Reads the restaurants store (Depends('restaurants')) for city listings.",
    ),
    "travel__get_restaurants_address": (
        "travel_restaurants",
        "Reads restaurant addresses from the restaurants store.",
    ),
    "travel__get_rating_reviews_for_restaurants": (
        "travel_restaurants",
        "Reads restaurant ratings/reviews from the restaurants store.",
    ),
    "travel__get_cuisine_type_for_restaurants": (
        "travel_restaurants",
        "Reads cuisine labels from the restaurants store.",
    ),
    "travel__get_dietary_restrictions_for_all_restaurants": (
        "travel_restaurants",
        "Reads dietary info from the restaurants store.",
    ),
    "travel__get_contact_information_for_restaurants": (
        "travel_restaurants",
        "Reads contact info from the restaurants store.",
    ),
    "travel__get_price_for_restaurants": (
        "travel_restaurants",
        "Reads pricing from the restaurants store.",
    ),
    "travel__check_restaurant_opening_hours": (
        "travel_restaurants",
        "Reads opening hours from the restaurants store.",
    ),
    # Car rentals (6)
    "travel__get_all_car_rental_companies_in_city": (
        "travel_car_rental",
        "Reads the car-rental store (Depends('car_rental')) for city listings.",
    ),
    "travel__get_car_types_available": (
        "travel_car_rental",
        "Reads available car types from the car-rental store.",
    ),
    "travel__get_rating_reviews_for_car_rental": (
        "travel_car_rental",
        "Reads ratings/reviews from the car-rental store.",
    ),
    "travel__get_car_rental_address": (
        "travel_car_rental",
        "Reads addresses from the car-rental store.",
    ),
    "travel__get_car_fuel_options": (
        "travel_car_rental",
        "Reads fuel options from the car-rental store.",
    ),
    "travel__get_car_price_per_day": (
        "travel_car_rental",
        "Reads daily rates from the car-rental store.",
    ),
    # Flights (1)
    "travel__get_flight_information": (
        "travel_flights",
        "Reads the flights store (Depends('flights')) for flight information.",
    ),
    # Calendar (2) -- the travel-side calendar store.
    "travel__get_day_calendar_events": (
        "travel_calendar",
        "Reads the travel-side calendar store (Depends('calendar')) for a day's "
        "events.",
    ),
    "travel__search_calendar_events": (
        "travel_calendar",
        "Reads the travel-side calendar store for matching events.",
    ),
}

_NAME_TO_TAG = {
    "banking_bank_account": BANKING_BANK_ACCOUNT_TAG,
    "banking_filesystem": BANKING_FILESYSTEM_TAG,
    "banking_user_account": BANKING_USER_ACCOUNT_TAG,
    "workspace_inbox": WORKSPACE_INBOX_TAG,
    "workspace_calendar": WORKSPACE_CALENDAR_TAG,
    "workspace_cloud_drive": WORKSPACE_CLOUD_DRIVE_TAG,
    "slack_slack": SLACK_SLACK_TAG,
    "slack_web": SLACK_WEB_TAG,
    "travel_user": TRAVEL_USER_TAG,
    "travel_hotels": TRAVEL_HOTELS_TAG,
    "travel_restaurants": TRAVEL_RESTAURANTS_TAG,
    "travel_car_rental": TRAVEL_CAR_RENTAL_TAG,
    "travel_flights": TRAVEL_FLIGHTS_TAG,
    "travel_calendar": TRAVEL_CALENDAR_TAG,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rationale_table_covers_every_read_tool() -> None:
    """Every read tool in READ_FUNCTION_NAMES must have a rationale entry."""
    missing = set(READ_FUNCTION_NAMES) - set(STORE_RATIONALES)
    assert not missing, (
        f"Read tools missing rationale entries: {sorted(missing)}.  "
        "Add a (store, rationale) row to STORE_RATIONALES."
    )


def test_no_stale_rationale_entries() -> None:
    """No rationale entry can reference a read tool that doesn't exist."""
    stale = set(STORE_RATIONALES) - set(READ_FUNCTION_NAMES)
    assert not stale, (
        f"STORE_RATIONALES references unknown read tools: {sorted(stale)}.  "
        "Either restore the tool in the registry or remove the entry."
    )


@pytest.mark.parametrize(
    "tool_name",
    sorted(STORE_RATIONALES.keys()),
    ids=lambda n: n.replace("__", ":"),
)
def test_each_tool_in_expected_store(tool_name: str) -> None:
    """For each tool, the live READ_STORE_MAP matches the rationale table."""
    expected_store_name, rationale = STORE_RATIONALES[tool_name]
    expected_tag = _NAME_TO_TAG[expected_store_name]
    actual_tag = READ_STORE_MAP[tool_name]
    assert actual_tag is expected_tag, (
        f"Store mismatch for {tool_name!r}:\n"
        f"  live READ_STORE_MAP -> {actual_tag.name!r}\n"
        f"  expected per rationale -> {expected_store_name!r}\n"
        f"  rationale: {rationale}\n"
        "Either update READ_STORE_MAP or update STORE_RATIONALES "
        "after revisiting the tool's Depends store."
    )


def test_every_rationale_store_is_a_known_leaf() -> None:
    """No typos in the expected_store_name strings."""
    for tool_name, (store_name, _) in STORE_RATIONALES.items():
        assert store_name in _NAME_TO_TAG, (
            f"Unknown store {store_name!r} on {tool_name!r}; "
            f"expected one of {sorted(_NAME_TO_TAG)}"
        )


def test_47_read_tools_total() -> None:
    """Sanity: AgentDojo v1 has exactly 47 readable tools across the
    four suites under our suite-prefix convention.  Changes here mean
    upstream added or removed a tool."""
    assert len(READ_FUNCTION_NAMES) == 47, (
        f"Read tool count is {len(READ_FUNCTION_NAMES)}, expected 47.  "
        "Upstream may have changed; revisit STORE_RATIONALES."
    )
    assert len(STORE_RATIONALES) == 47


def test_store_distribution() -> None:
    """Document and pin the per-store distribution of read tools.

    Sums to 47.  Two of the 16 store leaves (travel_reservation and
    travel_inbox) are write-only and so never appear here."""
    from collections import Counter

    counts = Counter(store for store, _ in STORE_RATIONALES.values())
    # If this distribution drifts, the store mapping has shifted.
    # Update both the assertion and the mapping commentary.
    assert dict(counts) == {
        "banking_bank_account": 4,
        "banking_filesystem": 1,
        "banking_user_account": 1,
        "workspace_inbox": 7,
        "workspace_calendar": 3,
        "workspace_cloud_drive": 4,
        "slack_slack": 4,
        "slack_web": 1,
        "travel_user": 1,
        "travel_hotels": 4,
        "travel_restaurants": 8,
        "travel_car_rental": 6,
        "travel_flights": 1,
        "travel_calendar": 2,
    }, f"Distribution drift: got {dict(counts)}.  Re-check assignments."
