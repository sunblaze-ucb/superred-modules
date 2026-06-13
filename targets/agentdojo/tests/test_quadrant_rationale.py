"""Per-tool quadrant rationale audit.

For every entry in :data:`agentdojo_target.controllables.READ_QUADRANT_MAP`,
this file pins:

- the expected 2x2 quadrant the tool sits in
- a one-line rationale grounded in the brief's content/data axis
  convention (content axis: who AUTHORED the data; data axis: who
  STORES/PROVIDES it; 1p = user / our system, 3p = external)

The parametrized test asserts that the live ``READ_QUADRANT_MAP`` exactly
matches this expected table.  A divergence either flags an upstream
change (new read tool) or a deliberate quadrant reassignment that the
author must mirror here.

This serves three purposes:

1. **Verification of the brief's Section 4.a table**: every spot-check
   in the brief is encoded here, plus the 35 tools the brief didn't
   enumerate.
2. **Living documentation**: ``QUADRANT_RATIONALES`` is the canonical
   one-line justification per tool.  Future readers see WHY a tool sits
   in its quadrant without having to derive it.
3. **Drift detection**: if a new tool is added to ``READ_FUNCTION_NAMES``
   without a rationale entry here, the test fails with the missing
   tool name.

When a tool can legitimately span quadrants (e.g. workspace search
emails covers both received and sent), we use the BROADER quadrant
(the one accessible to more attacker scopes).  This is the conservative
choice for an attacker-capability model: a scope that includes the
broader quadrant also reaches the tool; a narrower scope does not.
"""

from __future__ import annotations

import pytest

from agentdojo_target.controllables import READ_QUADRANT_MAP
from agentdojo_target.security_tags import (
    CONTENT_1P_DATA_1P_TAG,
    CONTENT_1P_DATA_3P_TAG,
    CONTENT_3P_DATA_1P_TAG,
    CONTENT_3P_DATA_3P_TAG,
)
from agentdojo_target.tool_registry import READ_FUNCTION_NAMES

# ---------------------------------------------------------------------------
# Per-tool (quadrant_name, rationale) table.
#
# Quadrant names are the .name strings of the SecurityDomainTag instances:
#   - "content_1p_data_1p"   (system clock, user-authored files in user-local FS)
#   - "content_1p_data_3p"   (user PII / drafts / scheduled txns held by 3p service)
#   - "content_3p_data_1p"   (3p-authored content in user-local storage, e.g. bills)
#   - "content_3p_data_3p"   (3p-authored content in 3p storage; the broadest leaf)
# ---------------------------------------------------------------------------

QUADRANT_RATIONALES: dict[str, tuple[str, str]] = {
    # ---- banking (6) ----
    "banking__get_iban": (
        "content_1p_data_3p",
        "User's own IBAN; the bank holds the identity record on a 3p ledger.",
    ),
    "banking__get_balance": (
        "content_1p_data_3p",
        "User's own balance; the bank holds the 3p ledger.",
    ),
    "banking__get_most_recent_transactions": (
        "content_3p_data_3p",
        "Transactions include subject fields authored by inbound counterparties "
        "(3p content), held by the bank (3p storage).  Broader quadrant.",
    ),
    "banking__get_scheduled_transactions": (
        "content_1p_data_3p",
        "User-initiated standing orders are 1p content; bank holds the schedule.",
    ),
    "banking__read_file": (
        "content_3p_data_1p",
        "Filesystem is user-local (1p storage), but bills / landlord notices / "
        "address-change letters are authored by 3p senders.",
    ),
    "banking__get_user_info": (
        "content_1p_data_3p",
        "User PII (name, address) is 1p content; the bank's customer record is 3p storage.",
    ),
    # ---- workspace (14) ----
    "workspace__search_emails": (
        "content_3p_data_3p",
        "Search hits include received emails from external senders (3p content); "
        "mail provider stores (3p).  Broader quadrant.",
    ),
    "workspace__get_sent_emails": (
        "content_1p_data_3p",
        "User-authored outbound messages (1p) on the mail provider's storage (3p).",
    ),
    "workspace__get_received_emails": (
        "content_3p_data_3p",
        "Inbound emails authored by external senders; provider stores.",
    ),
    "workspace__get_draft_emails": (
        "content_1p_data_3p",
        "User-authored drafts (1p) on the mail provider (3p).",
    ),
    "workspace__search_contacts_by_name": (
        "content_1p_data_3p",
        "User's curated address book (1p) on the mail provider (3p).",
    ),
    "workspace__search_contacts_by_email": (
        "content_1p_data_3p",
        "Same as search_contacts_by_name; user's address book on 3p storage.",
    ),
    "workspace__get_unread_emails": (
        "content_3p_data_3p",
        "Typically unread emails from external senders; same as get_received_emails.  "
        "Side-effect: flips read-flag (write-with-read-shape; see ASSUMPTIONS F.1).",
    ),
    "workspace__get_day_calendar_events": (
        "content_3p_data_3p",
        "Calendar events may have 3p-authored descriptions/invitees; provider stores.  "
        "Broader quadrant.",
    ),
    "workspace__search_calendar_events": (
        "content_3p_data_3p",
        "Same as get_day_calendar_events.",
    ),
    "workspace__get_current_day": (
        "content_1p_data_1p",
        "System clock value; no external dependency, deterministic local read.",
    ),
    "workspace__search_files_by_filename": (
        "content_1p_data_3p",
        "User's drive filenames (1p) on the provider's storage (3p).",
    ),
    "workspace__get_file_by_id": (
        "content_3p_data_3p",
        "Files include 3p-authored documents shared into the drive; provider stores.  "
        "Broader quadrant.",
    ),
    "workspace__list_files": (
        "content_1p_data_3p",
        "User's filename listing (1p) on the provider (3p).  Listing returns names "
        "only; content is reached via get_file_by_id which is broader.",
    ),
    "workspace__search_files": (
        "content_3p_data_3p",
        "Searches file CONTENT, which may include 3p-authored shared docs.  Broader.",
    ),
    # ---- slack (5) ----
    "slack__get_channels": (
        "content_3p_data_3p",
        "Channel names include External_* set by external admins (3p content); "
        "Slack stores (3p).",
    ),
    "slack__read_channel_messages": (
        "content_3p_data_3p",
        "Channel messages are mostly from external users; Slack stores.",
    ),
    "slack__read_inbox": (
        "content_3p_data_3p",
        "DMs from external users; Slack stores.",
    ),
    "slack__get_users_in_channel": (
        "content_3p_data_3p",
        "User list is managed by Slack; external admins can add users.",
    ),
    "slack__get_webpage": (
        "content_3p_data_3p",
        "External web content; the primary 3p/3p surface in the slack suite.",
    ),
    # ---- travel (22) ----
    "travel__get_user_information": (
        "content_1p_data_3p",
        "User PII (name, ID, passport, credit card, etc.) held by the travel "
        "agency (3p storage).  The primary 1p/3p surface in travel.",
    ),
    # Hotels (4)
    "travel__get_all_hotels_in_city": (
        "content_3p_data_3p",
        "3p hotel listings on the aggregator (3p storage).",
    ),
    "travel__get_hotels_prices": (
        "content_3p_data_3p",
        "Prices set by hotels (3p content); aggregator holds.",
    ),
    "travel__get_hotels_address": (
        "content_3p_data_3p",
        "Hotel addresses on the aggregator.",
    ),
    "travel__get_rating_reviews_for_hotels": (
        "content_3p_data_3p",
        "Reviews authored by 3p users; aggregator stores.  Primary injection surface "
        "in the travel suite (AgentDojo plants {injection_hotels_*} slots here).",
    ),
    # Restaurants (8)
    "travel__get_all_restaurants_in_city": (
        "content_3p_data_3p",
        "3p restaurant listings on the aggregator.",
    ),
    "travel__get_restaurants_address": (
        "content_3p_data_3p",
        "Restaurant addresses on the aggregator.",
    ),
    "travel__get_rating_reviews_for_restaurants": (
        "content_3p_data_3p",
        "Reviews authored by 3p users; secondary injection surface.",
    ),
    "travel__get_cuisine_type_for_restaurants": (
        "content_3p_data_3p",
        "Restaurant cuisine labels set by the restaurants (3p).",
    ),
    "travel__get_dietary_restrictions_for_all_restaurants": (
        "content_3p_data_3p",
        "Dietary info per restaurant; aggregator holds.",
    ),
    "travel__get_contact_information_for_restaurants": (
        "content_3p_data_3p",
        "Restaurant phone numbers; aggregator holds.",
    ),
    "travel__get_price_for_restaurants": (
        "content_3p_data_3p",
        "Restaurant pricing; aggregator holds.",
    ),
    "travel__check_restaurant_opening_hours": (
        "content_3p_data_3p",
        "Opening hours set by restaurants (3p).",
    ),
    # Car rentals (6)
    "travel__get_all_car_rental_companies_in_city": (
        "content_3p_data_3p",
        "Car rental listings; aggregator holds.",
    ),
    "travel__get_car_types_available": (
        "content_3p_data_3p",
        "Car-type catalog per rental company; aggregator holds.",
    ),
    "travel__get_rating_reviews_for_car_rental": (
        "content_3p_data_3p",
        "Reviews authored by 3p users; tertiary injection surface in the travel suite.",
    ),
    "travel__get_car_rental_address": (
        "content_3p_data_3p",
        "Car rental addresses on the aggregator.",
    ),
    "travel__get_car_fuel_options": (
        "content_3p_data_3p",
        "Fuel options per company.",
    ),
    "travel__get_car_price_per_day": (
        "content_3p_data_3p",
        "Daily rates set by car rental companies.",
    ),
    # Flights (1)
    "travel__get_flight_information": (
        "content_3p_data_3p",
        "Flight info on the aggregator; airlines author the data.",
    ),
    # Calendar (2) -- travel-side, same broad rule as workspace
    "travel__get_day_calendar_events": (
        "content_3p_data_3p",
        "Travel-side calendar may include 3p invitees; same broad rule as workspace.",
    ),
    "travel__search_calendar_events": (
        "content_3p_data_3p",
        "Same as travel get_day_calendar_events.",
    ),
}

_NAME_TO_TAG = {
    "content_1p_data_1p": CONTENT_1P_DATA_1P_TAG,
    "content_1p_data_3p": CONTENT_1P_DATA_3P_TAG,
    "content_3p_data_1p": CONTENT_3P_DATA_1P_TAG,
    "content_3p_data_3p": CONTENT_3P_DATA_3P_TAG,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rationale_table_covers_every_read_tool() -> None:
    """Every read tool in READ_FUNCTION_NAMES must have a rationale entry."""
    missing = set(READ_FUNCTION_NAMES) - set(QUADRANT_RATIONALES)
    assert not missing, (
        f"Read tools missing rationale entries: {sorted(missing)}.  "
        "Add a (quadrant, rationale) row to QUADRANT_RATIONALES."
    )


def test_no_stale_rationale_entries() -> None:
    """No rationale entry can reference a read tool that doesn't exist."""
    stale = set(QUADRANT_RATIONALES) - set(READ_FUNCTION_NAMES)
    assert not stale, (
        f"QUADRANT_RATIONALES references unknown read tools: {sorted(stale)}.  "
        "Either restore the tool in the registry or remove the entry."
    )


@pytest.mark.parametrize(
    "tool_name",
    sorted(QUADRANT_RATIONALES.keys()),
    ids=lambda n: n.replace("__", ":"),
)
def test_each_tool_in_expected_quadrant(tool_name: str) -> None:
    """For each tool, the live READ_QUADRANT_MAP matches the rationale table."""
    expected_quadrant_name, rationale = QUADRANT_RATIONALES[tool_name]
    expected_tag = _NAME_TO_TAG[expected_quadrant_name]
    actual_tag = READ_QUADRANT_MAP[tool_name]
    assert actual_tag is expected_tag, (
        f"Quadrant mismatch for {tool_name!r}:\n"
        f"  live READ_QUADRANT_MAP -> {actual_tag.name!r}\n"
        f"  expected per rationale -> {expected_quadrant_name!r}\n"
        f"  rationale: {rationale}\n"
        "Either update READ_QUADRANT_MAP or update QUADRANT_RATIONALES "
        "after revisiting the brief's content/data axis convention."
    )


def test_every_rationale_quadrant_is_a_known_leaf() -> None:
    """No typos in the expected_quadrant_name strings."""
    for tool_name, (quadrant_name, _) in QUADRANT_RATIONALES.items():
        assert quadrant_name in _NAME_TO_TAG, (
            f"Unknown quadrant {quadrant_name!r} on {tool_name!r}; "
            f"expected one of {sorted(_NAME_TO_TAG)}"
        )


def test_47_read_tools_total() -> None:
    """Sanity: AgentDojo v1 has exactly 47 readable tools across the
    four suites under our suite-prefix convention.  Changes here mean
    upstream added or removed a tool."""
    assert len(READ_FUNCTION_NAMES) == 47, (
        f"Read tool count is {len(READ_FUNCTION_NAMES)}, expected 47.  "
        "Upstream may have changed; revisit QUADRANT_RATIONALES."
    )
    assert len(QUADRANT_RATIONALES) == 47


def test_quadrant_distribution() -> None:
    """Document and pin the per-quadrant distribution of read tools."""
    from collections import Counter

    counts = Counter(quadrant for quadrant, _ in QUADRANT_RATIONALES.values())
    # If this distribution drifts, the threat-model balance has shifted.
    # Update both the assertion and the brief's mapping commentary.
    assert dict(counts) == {
        "content_1p_data_1p": 1,  # workspace get_current_day
        "content_1p_data_3p": 11,  # banking + workspace 1p reads
        "content_3p_data_1p": 1,  # banking read_file
        "content_3p_data_3p": 34,  # the broad majority (slack + travel + mixed reads)
    }, f"Distribution drift: got {dict(counts)}.  Re-check assignments."
