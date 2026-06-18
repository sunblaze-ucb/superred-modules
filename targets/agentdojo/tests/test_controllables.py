"""Tests for the Controllable enumeration."""

from __future__ import annotations

from agentdojo_target.controllables import (
    CONTROLLABLES,
    READ_CTRLS,
    READ_STORE_MAP,
    SYSTEM_PROMPT_CTRL,
    TOOL_CATALOG_CTRLS,
    USER_PROMPT_CTRL,
    WRITE_STORE_MAP,
)
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
    TOOLS_TAG,
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
from agentdojo_target.tool_registry import READ_FUNCTION_NAMES, WRITE_FUNCTION_NAMES

# The 16 per-store leaf tags, all of which live under TOOLS_TAG.
STORE_LEAVES = {
    BANKING_BANK_ACCOUNT_TAG,
    BANKING_FILESYSTEM_TAG,
    BANKING_USER_ACCOUNT_TAG,
    WORKSPACE_INBOX_TAG,
    WORKSPACE_CALENDAR_TAG,
    WORKSPACE_CLOUD_DRIVE_TAG,
    SLACK_SLACK_TAG,
    SLACK_WEB_TAG,
    TRAVEL_HOTELS_TAG,
    TRAVEL_RESTAURANTS_TAG,
    TRAVEL_CAR_RENTAL_TAG,
    TRAVEL_FLIGHTS_TAG,
    TRAVEL_USER_TAG,
    TRAVEL_CALENDAR_TAG,
    TRAVEL_RESERVATION_TAG,
    TRAVEL_INBOX_TAG,
}


def test_system_prompt_ctrl_is_on_prompt_tag() -> None:
    assert SYSTEM_PROMPT_CTRL.security_domain is SYSTEM_PROMPT_TAG


def test_user_prompt_ctrl_is_on_user_tag() -> None:
    assert USER_PROMPT_CTRL.security_domain is USER_TAG


def test_four_tool_catalog_ctrls() -> None:
    assert len(TOOL_CATALOG_CTRLS) == 4
    names = [c.name for c in TOOL_CATALOG_CTRLS]
    assert names == [
        "tool_catalog_register",
        "tool_catalog_replace",
        "tool_catalog_unregister",
        "tool_catalog_rewrite_doc",
    ]


def test_tool_catalog_ctrls_split_across_capabilities() -> None:
    """The four catalogue operations are partitioned across the three
    capability children of the grouping root: register -> add; replace and
    rewrite-doc -> edit; unregister -> remove.  Each capability is subsumed
    by the grouping root (subsumption tested separately)."""
    register, replace, unregister, rewrite = TOOL_CATALOG_CTRLS
    assert register.security_domain is TOOL_CATALOGUE_ADD_TAG
    assert replace.security_domain is TOOL_CATALOGUE_EDIT_TAG
    assert unregister.security_domain is TOOL_CATALOGUE_REMOVE_TAG
    assert rewrite.security_domain is TOOL_CATALOGUE_EDIT_TAG


def test_one_read_ctrl_per_read_tool() -> None:
    """Every read tool has exactly one matching Controllable."""
    assert set(READ_CTRLS) == set(READ_FUNCTION_NAMES)
    assert len(READ_CTRLS) == 47


def test_every_read_ctrl_is_on_a_store_leaf() -> None:
    """All per-read Controllables sit on one of the 16 per-store leaves,
    each of which is subsumed by the tools root."""
    for ctrl in READ_CTRLS.values():
        assert ctrl.security_domain in STORE_LEAVES
        # Sanity: the tools root subsumes every store leaf.
        assert TOOLS_TAG.includes(ctrl.security_domain)


def test_aggregate_controllables_no_duplicates() -> None:
    """The CONTROLLABLES list has unique names."""
    names = [c.name for c in CONTROLLABLES]
    assert len(names) == len(set(names))


def test_aggregate_controllables_size() -> None:
    """1 system_prompt + 1 user_prompt + 4 catalog + 47 reads = 53."""
    assert len(CONTROLLABLES) == 53


def test_read_store_map_values_are_store_leaves_under_tools() -> None:
    """Every read tool maps to a per-store leaf that lives under TOOLS_TAG,
    and the entries cover exactly the 47 read Controllables."""
    for tag in READ_STORE_MAP.values():
        assert tag in STORE_LEAVES
        assert TOOLS_TAG.includes(tag)
    assert set(READ_STORE_MAP) == set(READ_CTRLS)
    assert len(READ_STORE_MAP) == len(READ_CTRLS) == 47


def test_read_store_map_per_store_distribution() -> None:
    """Pin the per-store read distribution; the counts sum to 47."""
    counts: dict = {leaf: 0 for leaf in STORE_LEAVES}
    for tag in READ_STORE_MAP.values():
        counts[tag] += 1
    expected = {
        BANKING_BANK_ACCOUNT_TAG: 4,
        BANKING_FILESYSTEM_TAG: 1,
        BANKING_USER_ACCOUNT_TAG: 1,
        WORKSPACE_INBOX_TAG: 7,
        WORKSPACE_CALENDAR_TAG: 3,
        WORKSPACE_CLOUD_DRIVE_TAG: 4,
        SLACK_SLACK_TAG: 4,
        SLACK_WEB_TAG: 1,
        TRAVEL_USER_TAG: 1,
        TRAVEL_HOTELS_TAG: 4,
        TRAVEL_RESTAURANTS_TAG: 8,
        TRAVEL_CAR_RENTAL_TAG: 6,
        TRAVEL_FLIGHTS_TAG: 1,
        TRAVEL_CALENDAR_TAG: 2,
    }
    for leaf, n in expected.items():
        assert counts[leaf] == n
    # Leaves with no read tool stay at zero.
    for leaf in STORE_LEAVES - set(expected):
        assert counts[leaf] == 0
    assert sum(counts.values()) == len(READ_CTRLS) == 47


def test_write_store_map_covers_write_tools_on_store_leaves() -> None:
    """Writes get NO Controllable but are still mapped to the store leaf
    they mutate: WRITE_STORE_MAP covers exactly the write tools, and every
    value is a per-store leaf subsumed by TOOLS_TAG."""
    assert set(WRITE_STORE_MAP) == set(WRITE_FUNCTION_NAMES)
    for tag in WRITE_STORE_MAP.values():
        assert tag in STORE_LEAVES
        assert TOOLS_TAG.includes(tag)


def test_write_store_map_per_store_distribution() -> None:
    """Pin the per-store write distribution; the counts sum to 27."""
    counts: dict = {leaf: 0 for leaf in STORE_LEAVES}
    for tag in WRITE_STORE_MAP.values():
        counts[tag] += 1
    expected = {
        BANKING_BANK_ACCOUNT_TAG: 3,
        BANKING_USER_ACCOUNT_TAG: 2,
        WORKSPACE_INBOX_TAG: 2,
        WORKSPACE_CALENDAR_TAG: 4,
        WORKSPACE_CLOUD_DRIVE_TAG: 4,
        SLACK_SLACK_TAG: 5,
        SLACK_WEB_TAG: 1,
        TRAVEL_RESERVATION_TAG: 3,
        TRAVEL_CALENDAR_TAG: 2,
        TRAVEL_INBOX_TAG: 1,
    }
    for leaf, n in expected.items():
        assert counts[leaf] == n
    for leaf in STORE_LEAVES - set(expected):
        assert counts[leaf] == 0
    assert sum(counts.values()) == len(WRITE_STORE_MAP) == 27
