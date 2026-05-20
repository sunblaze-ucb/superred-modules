"""Tests for the suite-prefixed tool registry.

Verifies the registry contains exactly the upstream tools with the
right R/W classification, suite prefix, and rebound dependencies.
"""

from __future__ import annotations

import pytest
from agentdojo.functions_runtime import FunctionsRuntime

from agentdojo_target.env import CompositeEnvironment
from agentdojo_target.seed_loader import load_composite_seed
from agentdojo_target.tool_registry import (
    ALL_FUNCTIONS,
    READ_FUNCTION_NAMES,
    READ_TOOLS,
    SUITE_NAMES,
    TOOL_REGISTRY,
    WRITE_FUNCTION_NAMES,
    WRITE_TOOLS,
    prefixed_name,
    split_prefixed,
)


def test_registry_covers_all_four_suites() -> None:
    """Every entry's suite is one of the four known."""
    suites = {e.suite for e in TOOL_REGISTRY.values()}
    assert suites == set(SUITE_NAMES)


def test_expected_total_tool_count() -> None:
    """Per the explore-report inventory the v1 catalog has 74 tools."""
    expected = sum(len(READ_TOOLS[s]) + len(WRITE_TOOLS[s]) for s in SUITE_NAMES)
    assert len(TOOL_REGISTRY) == expected
    # Sanity: the inventory matches the report (11+24+11+28).
    assert len(READ_TOOLS["banking"]) + len(WRITE_TOOLS["banking"]) == 11
    assert len(READ_TOOLS["workspace"]) + len(WRITE_TOOLS["workspace"]) == 24
    assert len(READ_TOOLS["slack"]) + len(WRITE_TOOLS["slack"]) == 11
    assert len(READ_TOOLS["travel"]) + len(WRITE_TOOLS["travel"]) == 28


def test_read_and_write_sets_partition() -> None:
    """READ and WRITE are disjoint and together cover the whole registry."""
    assert READ_FUNCTION_NAMES & WRITE_FUNCTION_NAMES == set()
    assert READ_FUNCTION_NAMES | WRITE_FUNCTION_NAMES == set(TOOL_REGISTRY)


def test_prefixed_name_format() -> None:
    """Every name is ``{suite}__{tool}`` with the suite a known prefix."""
    for prefixed in TOOL_REGISTRY:
        suite, tool = split_prefixed(prefixed)
        assert suite in SUITE_NAMES
        assert prefixed == prefixed_name(suite, tool)


def test_split_prefixed_rejects_unknown_suite() -> None:
    with pytest.raises(ValueError, match="Unknown suite"):
        split_prefixed("nonsense__foo")


def test_split_prefixed_rejects_unprefixed() -> None:
    with pytest.raises(ValueError, match="not in"):
        split_prefixed("send_money")


def test_each_function_kind_matches_classification() -> None:
    """Per-entry kind matches its presence in READ/WRITE tables."""
    for entry in TOOL_REGISTRY.values():
        if entry.kind == "read":
            assert entry.original_name in READ_TOOLS[entry.suite]
            assert entry.original_name not in WRITE_TOOLS[entry.suite]
        else:
            assert entry.original_name in WRITE_TOOLS[entry.suite]
            assert entry.original_name not in READ_TOOLS[entry.suite]


def test_function_names_are_prefixed_at_runtime_level() -> None:
    """The renamed Function carries its prefix on its ``name`` field
    (so FunctionsRuntime.run_function dispatches on prefixed names)."""
    for entry in TOOL_REGISTRY.values():
        assert entry.function.name == entry.prefixed_name


def test_dependencies_rebound_to_composite() -> None:
    """Every Depends extractor pulls the matching sub-attribute from a
    composite environment, not a per-suite env."""
    env = load_composite_seed()
    for entry in TOOL_REGISTRY.values():
        for arg_name, dep in entry.function.dependencies.items():
            extracted = dep.extract_dep_from_env(env)
            # Confirm the extracted object is sourced from the right sub-env.
            sub = getattr(env, entry.suite)
            sub_attrs = {id(getattr(sub, a)) for a in type(sub).model_fields}
            assert id(extracted) in sub_attrs, (
                f"Dep {entry.prefixed_name}.{arg_name} extracted an object "
                f"that does not belong to env.{entry.suite}.*"
            )


def test_runtime_runs_a_prefixed_read_tool() -> None:
    """Smoke: register the prefixed Functions in a FunctionsRuntime and
    invoke a known read tool against a composite env, verifying the
    rebinding works end-to-end (no rewrite of the upstream tool body)."""
    env = load_composite_seed()
    runtime = FunctionsRuntime(ALL_FUNCTIONS)

    # banking__get_balance has no required user kwargs (signature is just
    # `account: Annotated[BankAccount, Depends("bank_account")]`).
    result, error = runtime.run_function(env, "banking__get_balance", {})
    assert error is None, error
    assert result == env.banking.bank_account.balance


def test_runtime_can_run_workspace_read() -> None:
    """Workspace tools, which share Inbox/Calendar classes with travel,
    still resolve to the workspace sub-env."""
    env = load_composite_seed()
    runtime = FunctionsRuntime(ALL_FUNCTIONS)

    # `workspace__get_current_day` reads `calendar.current_day` and
    # returns it serialised.  It must read the workspace calendar.
    result, error = runtime.run_function(env, "workspace__get_current_day", {})
    assert error is None, error
    assert result == env.workspace.calendar.current_day.isoformat()


def test_runtime_can_run_travel_read_without_workspace_contamination() -> None:
    """A travel calendar read must not pull from the workspace calendar."""
    env = load_composite_seed()
    runtime = FunctionsRuntime(ALL_FUNCTIONS)

    # Mutate the workspace calendar so the two are distinguishable.
    env.workspace.calendar.events.clear()
    assert env.travel.calendar.events, "Travel calendar should still be populated"

    # search_calendar_events takes a query; pass something that matches.
    result, error = runtime.run_function(
        env, "travel__search_calendar_events", {"query": "Tea"}
    )
    # The exact match content depends on the seed, but the call must not raise.
    assert error is None, f"Travel calendar read failed: {error}"


def test_no_workspace_into_travel_or_reverse() -> None:
    """Workspace and travel tools must rebind to their own sub-attribute."""
    workspace_inbox_tool = TOOL_REGISTRY["workspace__send_email"]
    travel_inbox_tool = TOOL_REGISTRY["travel__send_email"]
    env = load_composite_seed()

    ws_inbox_dep = workspace_inbox_tool.function.dependencies["inbox"]
    tv_inbox_dep = travel_inbox_tool.function.dependencies["inbox"]

    assert ws_inbox_dep.extract_dep_from_env(env) is env.workspace.inbox
    assert tv_inbox_dep.extract_dep_from_env(env) is env.travel.inbox
    assert env.workspace.inbox is not env.travel.inbox
