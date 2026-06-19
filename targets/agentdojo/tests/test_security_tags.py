"""Tests for the AgentDojo target's security-domain forest.

Verifies forest topology, parent-child relationships, scope-inclusion
semantics, and that the assembled :class:`SecurityDomain` exposes exactly
the expected roots.
"""

from __future__ import annotations

from superred.core.types.security_domain import scope_includes

from agentdojo_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TAG,
    BANKING_BANK_ACCOUNT_TAG,
    BANKING_FILESYSTEM_TAG,
    BANKING_TAG,
    BANKING_USER_ACCOUNT_TAG,
    DETAILED_SYSTEM_SPECIFICATION_TAG,
    DOMAIN,
    MODEL_IDENTITY_TAG,
    SLACK_SLACK_TAG,
    SLACK_TAG,
    SLACK_WEB_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_CATALOGUE_ADD_TAG,
    TOOL_CATALOGUE_EDIT_TAG,
    TOOL_CATALOGUE_REMOVE_TAG,
    TOOL_CATALOGUE_TAG,
    TOOLS_TAG,
    TRAVEL_CALENDAR_TAG,
    TRAVEL_CAR_RENTAL_TAG,
    TRAVEL_FLIGHTS_TAG,
    TRAVEL_HOTELS_TAG,
    TRAVEL_INBOX_TAG,
    TRAVEL_RESERVATION_TAG,
    TRAVEL_RESTAURANTS_TAG,
    TRAVEL_TAG,
    TRAVEL_USER_TAG,
    USER_TAG,
    WORKSPACE_CALENDAR_TAG,
    WORKSPACE_CLOUD_DRIVE_TAG,
    WORKSPACE_INBOX_TAG,
    WORKSPACE_TAG,
)

# Per-service grouping: each service node and the store leaves directly
# under it.  Used by the tools-subtree subsumption tests.
SERVICE_STORES = {
    BANKING_TAG: (
        BANKING_BANK_ACCOUNT_TAG,
        BANKING_FILESYSTEM_TAG,
        BANKING_USER_ACCOUNT_TAG,
    ),
    WORKSPACE_TAG: (
        WORKSPACE_INBOX_TAG,
        WORKSPACE_CALENDAR_TAG,
        WORKSPACE_CLOUD_DRIVE_TAG,
    ),
    SLACK_TAG: (
        SLACK_SLACK_TAG,
        SLACK_WEB_TAG,
    ),
    TRAVEL_TAG: (
        TRAVEL_HOTELS_TAG,
        TRAVEL_RESTAURANTS_TAG,
        TRAVEL_CAR_RENTAL_TAG,
        TRAVEL_FLIGHTS_TAG,
        TRAVEL_USER_TAG,
        TRAVEL_CALENDAR_TAG,
        TRAVEL_RESERVATION_TAG,
        TRAVEL_INBOX_TAG,
    ),
}


def test_domain_has_three_roots() -> None:
    """system, user, tools are the three independent trees."""
    root_names = {t.name for t in DOMAIN.roots}
    assert root_names == {"system", "user", "tools"}


def test_system_tag_includes_all_system_descendants() -> None:
    """Holding the system root grants every system-side capability."""
    for descendant in (
        SYSTEM_PROMPT_TAG,
        TOOL_CATALOGUE_TAG,
        TOOL_CATALOGUE_ADD_TAG,
        TOOL_CATALOGUE_EDIT_TAG,
        TOOL_CATALOGUE_REMOVE_TAG,
        MODEL_IDENTITY_TAG,
        DETAILED_SYSTEM_SPECIFICATION_TAG,
        AGENT_TRACE_TAG,
        AGENT_TRACE_MESSAGES_TAG,
    ):
        assert SYSTEM_TAG.includes(descendant), descendant.name


def test_tool_catalogue_subsumption() -> None:
    """The catalogue grouping root subsumes its three capability children;
    none of the children subsumes the root or a sibling capability."""
    for child in (
        TOOL_CATALOGUE_ADD_TAG,
        TOOL_CATALOGUE_EDIT_TAG,
        TOOL_CATALOGUE_REMOVE_TAG,
    ):
        assert TOOL_CATALOGUE_TAG.includes(child), child.name
        # a child capability does not imply the broad grouping root
        assert not child.includes(TOOL_CATALOGUE_TAG), child.name
    # the three capabilities are independent siblings: granting one (e.g.
    # edit) does not imply another (e.g. add).
    assert not TOOL_CATALOGUE_EDIT_TAG.includes(TOOL_CATALOGUE_ADD_TAG)
    assert not TOOL_CATALOGUE_ADD_TAG.includes(TOOL_CATALOGUE_EDIT_TAG)
    assert not TOOL_CATALOGUE_REMOVE_TAG.includes(TOOL_CATALOGUE_ADD_TAG)


def test_agent_trace_subsumption() -> None:
    """Aggregate agent_trace subsumes its sole messages child.

    Tool calls and tool responses are no longer mirrored on the trace;
    each tool's call and return live once on its ControllablePostCallEvent.
    """
    assert AGENT_TRACE_TAG.includes(AGENT_TRACE_MESSAGES_TAG)


def test_user_has_no_children() -> None:
    """The user tree is a single tag with no internal hierarchy."""
    for tag in (
        SYSTEM_TAG,
        TOOLS_TAG,
        SYSTEM_PROMPT_TAG,
        MODEL_IDENTITY_TAG,
        AGENT_TRACE_TAG,
        BANKING_BANK_ACCOUNT_TAG,
    ):
        assert not USER_TAG.includes(tag)


def test_tools_root_subsumes_every_service_and_store() -> None:
    """The grouping root subsumes all four services and every store leaf."""
    for service, stores in SERVICE_STORES.items():
        assert TOOLS_TAG.includes(service), service.name
        for store in stores:
            assert TOOLS_TAG.includes(store), store.name


def test_service_subsumes_own_stores() -> None:
    """Each service node subsumes exactly the store leaves below it."""
    for service, stores in SERVICE_STORES.items():
        for store in stores:
            assert service.includes(store), (
                f"{service.name} should include {store.name}"
            )
            # a store leaf does not subsume its parent service
            assert not store.includes(service), (
                f"{store.name} should not include {service.name}"
            )


def test_service_does_not_subsume_other_services_stores() -> None:
    """A service does not include another service's store leaves."""
    for service, stores in SERVICE_STORES.items():
        for other_service, other_stores in SERVICE_STORES.items():
            if other_service is service:
                continue
            for other_store in other_stores:
                assert not service.includes(other_store), (
                    f"{service.name} should not include {other_store.name}"
                )


def test_sibling_services_do_not_subsume_each_other() -> None:
    """The four service nodes are siblings; none subsumes another."""
    services = list(SERVICE_STORES)
    for a in services:
        for b in services:
            if a is not b:
                assert not a.includes(b), f"{a.name} should not include {b.name}"


def test_sibling_stores_do_not_subsume_each_other() -> None:
    """Store leaves never subsume one another, within or across services."""
    all_stores = [store for stores in SERVICE_STORES.values() for store in stores]
    for a in all_stores:
        for b in all_stores:
            if a is not b:
                assert not a.includes(b), f"{a.name} should not include {b.name}"


def test_workspace_and_travel_calendars_are_distinct() -> None:
    """The workspace and travel calendars are separate store instances:
    holding one grants no access to the other (and likewise for the two
    inboxes the two suites each own)."""
    assert WORKSPACE_CALENDAR_TAG is not TRAVEL_CALENDAR_TAG
    assert not WORKSPACE_CALENDAR_TAG.includes(TRAVEL_CALENDAR_TAG)
    assert not TRAVEL_CALENDAR_TAG.includes(WORKSPACE_CALENDAR_TAG)
    assert not WORKSPACE_INBOX_TAG.includes(TRAVEL_INBOX_TAG)
    assert not TRAVEL_INBOX_TAG.includes(WORKSPACE_INBOX_TAG)


def test_independent_trees_do_not_cross() -> None:
    """Roots do not include tags from other trees."""
    assert not SYSTEM_TAG.includes(USER_TAG)
    assert not SYSTEM_TAG.includes(TOOLS_TAG)
    assert not USER_TAG.includes(SYSTEM_TAG)
    assert not USER_TAG.includes(TOOLS_TAG)
    assert not TOOLS_TAG.includes(SYSTEM_TAG)
    assert not TOOLS_TAG.includes(USER_TAG)


def test_scope_includes_uses_tags() -> None:
    """Sanity check that scope_includes() agrees with tag.includes() across roots."""
    scope = frozenset({SYSTEM_TAG})
    assert scope_includes(scope, SYSTEM_PROMPT_TAG)
    assert scope_includes(scope, AGENT_TRACE_MESSAGES_TAG)
    assert not scope_includes(scope, USER_TAG)
    assert not scope_includes(scope, BANKING_BANK_ACCOUNT_TAG)


def test_assembled_domain_size() -> None:
    """The assembled DOMAIN has the three roots and 32 total tags."""
    assert {t.name for t in DOMAIN.roots} == {"system", "user", "tools"}
    # Count tags directly from the forest map.  Enumerating
    # distinct_combinations() to count would be exponential in the tools subtree
    # (tens of millions of antichains -> hundreds of seconds) and needless for a
    # count; distinct_combinations correctness is covered by the small-subtree
    # test in the asb target.
    assert len(DOMAIN._tags) == 32


def test_assembled_domain_tag_count() -> None:
    """The 32 tags split system 10, user 1, tools 21 across the three trees."""
    by_root: dict[str, int] = {"system": 0, "user": 0, "tools": 0}
    for tag in DOMAIN._tags.values():
        node = tag
        while node.parent is not None:
            node = node.parent
        by_root[node.name] += 1
    assert by_root == {"system": 10, "user": 1, "tools": 21}
