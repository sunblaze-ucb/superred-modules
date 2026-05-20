"""Module-level Controllable singletons exposed by :class:`AgentDojoTarget`.

Categories:

- ``SYSTEM_PROMPT_CTRL``: agent system prompt (scope ``system.prompt``).
- ``USER_PROMPT_CTRL``: benign user instruction (scope ``user``).  An
  attacker with the ``user`` tag in scope can override the Task's
  benign prompt via :class:`ControllableInjection`.
- Tool-catalogue controllables: four operations partitioned across
  ``system.tool_catalogue`` (broad write) and
  ``system.tool_catalogue_addable`` (register-only).
- Per-read controllables: one per readable tool in the catalog
  (47 entries), each mapped to one leaf of the
  ``tools`` 2x2 quadrant grid.

The 2x2 quadrant mapping for each read tool lives in
:data:`READ_QUADRANT_MAP`; see ``ASSUMPTIONS.md`` §C.4 for the rationale
behind each assignment.  Where a tool can legitimately span quadrants
(e.g. ``workspace.search_emails`` searches both received and sent), we
use the *broader* quadrant to remain conservative on the attacker side:
the broader scope is needed for the tool to be reachable.
"""

from __future__ import annotations

from superred.core.types.controllable import Controllable

from agentdojo_target.security_tags import (
    CONTENT_1P_DATA_1P_TAG,
    CONTENT_1P_DATA_3P_TAG,
    CONTENT_3P_DATA_1P_TAG,
    CONTENT_3P_DATA_3P_TAG,
    PROMPT_TAG,
    TOOL_CATALOGUE_ADDABLE_TAG,
    TOOL_CATALOGUE_TAG,
    USER_TAG,
)
from agentdojo_target.tool_registry import (
    READ_FUNCTION_NAMES,
    SUITE_NAMES,
    TOOL_REGISTRY,
    prefixed_name,
    split_prefixed,
)

# ---------------------------------------------------------------------------
# Prompt controllables
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_CTRL: Controllable = Controllable(
    name="system_prompt",
    security_domain=PROMPT_TAG,
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
    security_domain=TOOL_CATALOGUE_ADDABLE_TAG,
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
    security_domain=TOOL_CATALOGUE_TAG,
    description=(
        "Replace an existing tool (shadow attack): when the agent calls "
        'the named tool, the attacker-supplied fake_return is used. '
        'Injection value: {"name": str, "fake_return": Any, '
        '"description"?: str}.'
    ),
    value_type="json",
)

TOOL_CATALOG_UNREGISTER_CTRL: Controllable = Controllable(
    name="tool_catalog_unregister",
    security_domain=TOOL_CATALOGUE_TAG,
    description=(
        "Remove an existing tool from the catalog.  Injection value: "
        '{"name": str}.'
    ),
    value_type="json",
)

TOOL_CATALOG_REWRITE_DOC_CTRL: Controllable = Controllable(
    name="tool_catalog_rewrite_doc",
    security_domain=TOOL_CATALOGUE_TAG,
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
# Per-read 2x2 quadrant mapping
#
# Convention recap (security_tags.py):
#   content axis: who AUTHORED the content the tool returns
#   data axis:    who STORES / PROVIDES ACCESS to the data
# 1p = first-party (user / our system); 3p = third-party (external).
#
# When a tool legitimately spans quadrants (e.g. workspace.search_emails
# searches both received and sent), we use the broader quadrant so the
# tool is reachable to any attacker who could legitimately observe the
# wider content/storage class.
# ---------------------------------------------------------------------------

READ_QUADRANT_MAP: dict[str, object] = {
    # ---- banking ----
    "banking__get_iban":                    CONTENT_1P_DATA_3P_TAG,
    "banking__get_balance":                 CONTENT_1P_DATA_3P_TAG,
    # Transactions: subject field can be set by counterparty on inbound,
    # so the BROAD quadrant is 3p/3p.
    "banking__get_most_recent_transactions": CONTENT_3P_DATA_3P_TAG,
    "banking__get_scheduled_transactions":  CONTENT_1P_DATA_3P_TAG,
    # File reads can return both user-written files and 3p-authored letters
    # (bills, landlord notices, address-change letters).  Broad: 3p content
    # in 1p storage (the user's filesystem).
    "banking__read_file":                   CONTENT_3P_DATA_1P_TAG,
    "banking__get_user_info":               CONTENT_1P_DATA_3P_TAG,

    # ---- workspace ----
    # Emails: search/received covers external senders -> broad is 3p/3p.
    "workspace__search_emails":             CONTENT_3P_DATA_3P_TAG,
    "workspace__get_sent_emails":           CONTENT_1P_DATA_3P_TAG,
    "workspace__get_received_emails":       CONTENT_3P_DATA_3P_TAG,
    "workspace__get_draft_emails":          CONTENT_1P_DATA_3P_TAG,
    "workspace__search_contacts_by_name":   CONTENT_1P_DATA_3P_TAG,
    "workspace__search_contacts_by_email":  CONTENT_1P_DATA_3P_TAG,
    "workspace__get_unread_emails":         CONTENT_3P_DATA_3P_TAG,
    # Calendar: events authored by external invitees count as 3p content.
    "workspace__get_day_calendar_events":   CONTENT_3P_DATA_3P_TAG,
    "workspace__search_calendar_events":    CONTENT_3P_DATA_3P_TAG,
    "workspace__get_current_day":           CONTENT_1P_DATA_1P_TAG,
    # Cloud drive: list/search-by-name surfaces user file names (1p content
    # in 3p storage); content reads can return shared 3p files (broad).
    "workspace__search_files_by_filename":  CONTENT_1P_DATA_3P_TAG,
    "workspace__get_file_by_id":            CONTENT_3P_DATA_3P_TAG,
    "workspace__list_files":                CONTENT_1P_DATA_3P_TAG,
    "workspace__search_files":              CONTENT_3P_DATA_3P_TAG,

    # ---- slack ----
    "slack__get_channels":                  CONTENT_3P_DATA_3P_TAG,
    "slack__read_channel_messages":         CONTENT_3P_DATA_3P_TAG,
    "slack__read_inbox":                    CONTENT_3P_DATA_3P_TAG,
    "slack__get_users_in_channel":          CONTENT_3P_DATA_3P_TAG,
    "slack__get_webpage":                   CONTENT_3P_DATA_3P_TAG,

    # ---- travel ----
    "travel__get_user_information":         CONTENT_1P_DATA_3P_TAG,
    # Hotels (4)
    "travel__get_all_hotels_in_city":           CONTENT_3P_DATA_3P_TAG,
    "travel__get_hotels_prices":                CONTENT_3P_DATA_3P_TAG,
    "travel__get_hotels_address":               CONTENT_3P_DATA_3P_TAG,
    "travel__get_rating_reviews_for_hotels":    CONTENT_3P_DATA_3P_TAG,
    # Restaurants (8)
    "travel__get_all_restaurants_in_city":              CONTENT_3P_DATA_3P_TAG,
    "travel__get_restaurants_address":                  CONTENT_3P_DATA_3P_TAG,
    "travel__get_rating_reviews_for_restaurants":       CONTENT_3P_DATA_3P_TAG,
    "travel__get_cuisine_type_for_restaurants":         CONTENT_3P_DATA_3P_TAG,
    "travel__get_dietary_restrictions_for_all_restaurants": CONTENT_3P_DATA_3P_TAG,
    "travel__get_contact_information_for_restaurants":  CONTENT_3P_DATA_3P_TAG,
    "travel__get_price_for_restaurants":                CONTENT_3P_DATA_3P_TAG,
    "travel__check_restaurant_opening_hours":           CONTENT_3P_DATA_3P_TAG,
    # Car rentals (6)
    "travel__get_all_car_rental_companies_in_city": CONTENT_3P_DATA_3P_TAG,
    "travel__get_car_types_available":              CONTENT_3P_DATA_3P_TAG,
    "travel__get_rating_reviews_for_car_rental":    CONTENT_3P_DATA_3P_TAG,
    "travel__get_car_rental_address":               CONTENT_3P_DATA_3P_TAG,
    "travel__get_car_fuel_options":                 CONTENT_3P_DATA_3P_TAG,
    "travel__get_car_price_per_day":                CONTENT_3P_DATA_3P_TAG,
    # Flights (1)
    "travel__get_flight_information":               CONTENT_3P_DATA_3P_TAG,
    # Calendar (2) -- the travel calendar is the user's, but events may be
    # authored by third parties (the same broad rule as workspace).
    "travel__get_day_calendar_events":              CONTENT_3P_DATA_3P_TAG,
    "travel__search_calendar_events":               CONTENT_3P_DATA_3P_TAG,
}


# ---------------------------------------------------------------------------
# Build per-read Controllables
# ---------------------------------------------------------------------------


def _make_read_ctrl(prefixed: str) -> Controllable:
    """Build a Controllable for a single prefixed read-tool name."""
    if prefixed not in READ_QUADRANT_MAP:
        raise RuntimeError(
            f"Read tool {prefixed!r} has no entry in READ_QUADRANT_MAP. "
            "Add an entry classifying it into one of the four 2x2 leaves."
        )
    suite, original = split_prefixed(prefixed)
    return Controllable(
        name=f"read__{prefixed}",
        security_domain=READ_QUADRANT_MAP[prefixed],
        description=(
            f"Per-read injection point for ``{prefixed}`` "
            f"(suite={suite}, upstream tool={original}).  When the agent "
            "invokes this read, the runtime fires a "
            "ControllablePostCallEvent carrying the legitimate value as "
            "``answer``; a ControllableInjection response replaces the "
            "agent-visible return."
        ),
        value_type="json",
    )


def _build_read_controllables() -> dict[str, Controllable]:
    """Build one Controllable per read tool, in deterministic order.

    Validates that :data:`READ_QUADRANT_MAP` covers every read tool and
    contains no extras (no stale mappings).
    """
    mapped = set(READ_QUADRANT_MAP)
    expected = set(READ_FUNCTION_NAMES)
    missing = expected - mapped
    if missing:
        raise RuntimeError(
            "READ_QUADRANT_MAP is missing entries for read tools: "
            f"{sorted(missing)}"
        )
    stale = mapped - expected
    if stale:
        raise RuntimeError(
            "READ_QUADRANT_MAP has entries for non-read tools: "
            f"{sorted(stale)}"
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
    "READ_QUADRANT_MAP",
    "READ_CTRLS",
    "CONTROLLABLES",
]
