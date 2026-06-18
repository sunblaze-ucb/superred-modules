"""Unit tests for the redesigned trust-boundary forest (tools > scenario > tool)."""

from __future__ import annotations

import json
from pathlib import Path

from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag, scope_includes

from asb_target import security_tags as st
from asb_target.security_tags import (
    AGENT_TRACE_TAG,
    DETAILED_SYSTEM_SPECIFICATION_TAG,
    DOMAIN,
    MEMORY_TAG,
    MODEL_IDENTITY_TAG,
    NORMAL_TOOL_NAMES,
    SCENARIO_TOOL_TAGS,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_CATALOGUE_ADD_TAG,
    TOOL_CATALOGUE_EDIT_TAG,
    TOOL_CATALOGUE_REMOVE_TAG,
    TOOL_CATALOGUE_TAG,
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
        MODEL_IDENTITY_TAG,
        DETAILED_SYSTEM_SPECIFICATION_TAG,
        TOOL_CATALOGUE_TAG,
        TOOL_CATALOGUE_ADD_TAG,
        TOOL_CATALOGUE_EDIT_TAG,
        TOOL_CATALOGUE_REMOVE_TAG,
        TOOLS_TAG,
        MEMORY_TAG,
    ]
    tags.extend(SCENARIO_TOOL_TAGS.values())
    tags.extend(TOOL_OBSERVATION_TAGS.values())
    return tags


def test_four_roots_present() -> None:
    assert {r.name for r in DOMAIN.roots} == {"user", "system", "tools", "memory"}


def test_no_removed_tags() -> None:
    names = {t.name for t in _all_tags()}
    forbidden = {
        "external",
        "internal",
        "external_tool_observation",
        "internal_tool_observation",
        "tool_catalog_readable",
        "user_readable",
        "system_prompt_readable",
        "memory_readable",
    }
    assert not (names & forbidden)
    assert not any(n.endswith("_readable") for n in names)


def test_tools_tree_is_three_levels_per_tool() -> None:
    assert len(NORMAL_TOOL_NAMES) == 20
    assert len(SCENARIO_TOOL_TAGS) == 10  # scenario nodes
    assert len(TOOL_OBSERVATION_TAGS) == 20  # one leaf per tool (fully representative)
    # every tool has a distinct leaf whose parent is its scenario node
    leaves = list(TOOL_OBSERVATION_TAGS.values())
    assert len(set(id(t) for t in leaves)) == 20
    for scenario, tools in TOOLS_BY_SCENARIO.items():
        scenario_tag = SCENARIO_TOOL_TAGS[scenario]
        assert scenario_tag.parent is TOOLS_TAG
        for tool in tools:
            leaf = TOOL_OBSERVATION_TAGS[tool]
            assert leaf.parent is scenario_tag
            assert leaf.name == f"tools.{scenario}.{tool}"


def test_subsumption_at_every_level() -> None:
    tools_scope = frozenset({TOOLS_TAG})
    # {tools} subsumes every scenario node and every tool leaf
    for tag in list(SCENARIO_TOOL_TAGS.values()) + list(TOOL_OBSERVATION_TAGS.values()):
        assert scope_includes(tools_scope, tag)
    # a scenario scope subsumes its own tools but not another scenario's
    fin = SCENARIO_TOOL_TAGS["financial_analyst"]
    fin_scope = frozenset({fin})
    assert scope_includes(fin_scope, TOOL_OBSERVATION_TAGS["market_data_api"])
    assert scope_includes(fin_scope, TOOL_OBSERVATION_TAGS["portfolio_manager"])
    assert not scope_includes(fin_scope, TOOL_OBSERVATION_TAGS["sys_monitor"])
    assert not scope_includes(fin_scope, TOOLS_TAG)
    # a single-tool scope is the narrowest: only that tool
    leaf_scope = frozenset({TOOL_OBSERVATION_TAGS["market_data_api"]})
    assert scope_includes(leaf_scope, TOOL_OBSERVATION_TAGS["market_data_api"])
    assert not scope_includes(leaf_scope, TOOL_OBSERVATION_TAGS["portfolio_manager"])
    assert not scope_includes(leaf_scope, fin)


def test_system_subsumes_its_children() -> None:
    system_scope = frozenset({SYSTEM_TAG})
    for child in (
        SYSTEM_PROMPT_TAG,
        AGENT_TRACE_TAG,
        MODEL_IDENTITY_TAG,
        DETAILED_SYSTEM_SPECIFICATION_TAG,
        TOOL_CATALOGUE_TAG,
        TOOL_CATALOGUE_ADD_TAG,
        TOOL_CATALOGUE_EDIT_TAG,
        TOOL_CATALOGUE_REMOVE_TAG,
    ):
        assert scope_includes(system_scope, child)
    # tool_catalogue is a grouping root over its three edit capabilities
    cat_scope = frozenset({TOOL_CATALOGUE_TAG})
    for child in (TOOL_CATALOGUE_ADD_TAG, TOOL_CATALOGUE_EDIT_TAG, TOOL_CATALOGUE_REMOVE_TAG):
        assert scope_includes(cat_scope, child)
    # a single edit capability does not subsume its siblings
    assert not scope_includes(frozenset({TOOL_CATALOGUE_EDIT_TAG}), TOOL_CATALOGUE_ADD_TAG)
    assert not scope_includes(frozenset({TOOL_CATALOGUE_ADD_TAG}), TOOL_CATALOGUE_REMOVE_TAG)


def test_tool_catalogue_is_separate_from_tool_interactions() -> None:
    # The fixed flaw: reading the tool-catalogue LISTING (which tools exist) must
    # NOT imply reading every tool's returned observation. The listing is tagged
    # at tool_catalogue (a system child); tool interactions live under the
    # independent tools root. So tool_catalogue is NOT an ancestor of any tool
    # node, and {tools} does not grant the catalogue capability.
    cat_scope = frozenset({TOOL_CATALOGUE_TAG})
    assert not scope_includes(cat_scope, TOOLS_TAG)
    for scen in SCENARIO_TOOL_TAGS.values():
        assert not scope_includes(cat_scope, scen)
    for leaf in TOOL_OBSERVATION_TAGS.values():
        assert not scope_includes(cat_scope, leaf)
    assert not scope_includes(frozenset({TOOLS_TAG}), TOOL_CATALOGUE_TAG)


def test_model_identity_is_a_read_only_system_child() -> None:
    assert MODEL_IDENTITY_TAG.parent is SYSTEM_TAG
    # knowing the model is independent of any write capability
    mi_scope = frozenset({MODEL_IDENTITY_TAG})
    assert not scope_includes(mi_scope, SYSTEM_PROMPT_TAG)
    assert not scope_includes(mi_scope, TOOL_CATALOGUE_TAG)
    assert not scope_includes(mi_scope, AGENT_TRACE_TAG)


def test_agent_trace_holds_only_agent_generations_not_tool_data() -> None:
    # A whole tool interaction (call + return) is the tool's data and must be
    # reachable only with that tool's scope, never via {system}. agent_trace is a
    # leaf for the agent's OWN generations only: no tool-call or tool-response tag.
    names = {t.name for t in _all_tags()}
    assert "agent_trace_tool_calls" not in names
    assert "agent_trace_tool_responses" not in names
    assert not [t for t in _all_tags() if t.parent is AGENT_TRACE_TAG]  # agent_trace is a leaf
    # a {system} attacker cannot read any tool's interaction surface
    system_scope = frozenset({SYSTEM_TAG})
    for leaf in TOOL_OBSERVATION_TAGS.values():
        assert not scope_includes(system_scope, leaf)


def test_tools_by_scenario_matches_dataset() -> None:
    # The forest's scenario->tool map is provably the real ASB dataset, not an
    # assumed regular pattern; this guards against silent drift.
    data = Path(st.__file__).parent / "data" / "all_normal_tools.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    by_agent: dict[str, list[str]] = {}
    for r in rows:
        by_agent.setdefault(r["Corresponding Agent"], []).append(r["Tool Name"])
    expected = {agent[: -len("_agent")]: tuple(tools) for agent, tools in by_agent.items()}
    assert TOOLS_BY_SCENARIO == expected
    assert all(len(tools) == 2 for tools in TOOLS_BY_SCENARIO.values())


def test_memory_is_its_own_root() -> None:
    assert not scope_includes(frozenset({TOOLS_TAG}), MEMORY_TAG)
    assert not scope_includes(frozenset({SYSTEM_TAG}), MEMORY_TAG)
    assert scope_includes(frozenset({MEMORY_TAG}), MEMORY_TAG)


def test_distinct_combinations_on_a_subtree() -> None:
    # distinct_combinations is exponential in child count, so it is impractical
    # on the full per-tool forest (and is never called on the run path, which
    # uses scope_includes). Verify the antichain enumeration is correct on a
    # small tools > scenario > tool subtree.
    root = SecurityDomainTag("tools")
    scen = SecurityDomainTag("tools.s", parent=root)
    t1 = SecurityDomainTag("tools.s.a", parent=scen)
    t2 = SecurityDomainTag("tools.s.b", parent=scen)
    combos = SecurityDomain([root, scen, t1, t2]).distinct_combinations()
    combos_set = {frozenset(c) for c in combos}
    # antichains: {}, {root}, {scen}, {t1}, {t2}, {t1,t2}
    assert frozenset({root}) in combos_set
    assert frozenset({scen}) in combos_set
    assert frozenset({t1, t2}) in combos_set
    assert frozenset({root, scen}) not in combos_set  # root subsumes scen -> not an antichain
