"""Unit tests for the redesigned trust-boundary forest."""

from __future__ import annotations

import time

from superred.core.types.security_domain import scope_includes

from asb_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    AGENT_TRACE_TOOL_RESPONSES_TAG,
    DOMAIN,
    MEMORY_TAG,
    NORMAL_TOOL_NAMES,
    SCENARIO_TOOL_TAGS,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_OBSERVATION_TAGS,
    TOOLS_BY_SCENARIO,
    TOOLS_TAG,
    USER_TAG,
)


def _all_tags() -> list:
    tags = [
        USER_TAG,
        SYSTEM_TAG,
        SYSTEM_PROMPT_TAG,
        AGENT_TRACE_TAG,
        AGENT_TRACE_MESSAGES_TAG,
        AGENT_TRACE_TOOL_CALLS_TAG,
        AGENT_TRACE_TOOL_RESPONSES_TAG,
        TOOLS_TAG,
        MEMORY_TAG,
    ]
    tags.extend(SCENARIO_TOOL_TAGS.values())
    return tags


def test_four_roots_present() -> None:
    root_names = {r.name for r in DOMAIN.roots}
    assert root_names == {"user", "system", "tools", "memory"}


def test_no_removed_tags() -> None:
    names = {t.name for t in _all_tags()}
    forbidden = {
        "external",
        "internal",
        "external_tool_observation",
        "internal_tool_observation",
        "model_identity",
        "tool_catalog_readable",
        "user_readable",
        "system_prompt_readable",
        "memory_readable",
    }
    assert not (names & forbidden), f"removed tags still present: {names & forbidden}"
    assert not any(n.endswith("_readable") for n in names)


def test_one_scenario_leaf_per_scenario() -> None:
    assert len(NORMAL_TOOL_NAMES) == 20
    assert len(SCENARIO_TOOL_TAGS) == 10
    assert len(TOOLS_BY_SCENARIO) == 10
    # every tool maps to a scenario tag
    assert set(TOOL_OBSERVATION_TAGS) == set(NORMAL_TOOL_NAMES)
    # the two tools of a scenario share that scenario's tag
    for scenario, tools in TOOLS_BY_SCENARIO.items():
        tag = SCENARIO_TOOL_TAGS[scenario]
        for tool in tools:
            assert TOOL_OBSERVATION_TAGS[tool] is tag


def test_tools_root_subsumes_every_scenario() -> None:
    tools_scope = frozenset({TOOLS_TAG})
    for tag in SCENARIO_TOOL_TAGS.values():
        assert scope_includes(tools_scope, tag)


def test_single_scenario_scope_is_narrow() -> None:
    fin = SCENARIO_TOOL_TAGS["financial_analyst"]
    admin = SCENARIO_TOOL_TAGS["system_admin"]
    assert scope_includes(frozenset({fin}), fin)
    assert not scope_includes(frozenset({fin}), admin)
    assert not scope_includes(frozenset({fin}), TOOLS_TAG)


def test_system_subsumes_prompt_and_trace() -> None:
    system_scope = frozenset({SYSTEM_TAG})
    for child in (
        SYSTEM_PROMPT_TAG,
        AGENT_TRACE_TAG,
        AGENT_TRACE_MESSAGES_TAG,
        AGENT_TRACE_TOOL_CALLS_TAG,
        AGENT_TRACE_TOOL_RESPONSES_TAG,
    ):
        assert scope_includes(system_scope, child)


def test_memory_is_its_own_root() -> None:
    assert not scope_includes(frozenset({TOOLS_TAG}), MEMORY_TAG)
    assert not scope_includes(frozenset({SYSTEM_TAG}), MEMORY_TAG)
    assert scope_includes(frozenset({MEMORY_TAG}), MEMORY_TAG)


def test_distinct_combinations_feasible_and_cross_channel() -> None:
    start = time.perf_counter()
    combos = DOMAIN.distinct_combinations()
    elapsed = time.perf_counter() - start
    assert combos
    # must stay cheap (the child-count is bounded so the antichain count does
    # not explode) -- this guards against a future per-tool-leaf regression.
    assert elapsed < 2.0, f"distinct_combinations too slow ({elapsed:.2f}s)"
    fin = SCENARIO_TOOL_TAGS["financial_analyst"]
    assert any(
        scope_includes(frozenset(c), USER_TAG) and scope_includes(frozenset(c), fin) for c in combos
    )
