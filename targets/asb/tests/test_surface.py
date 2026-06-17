"""Unit tests for the target's exposed surface: controllables, config slots,
observables, and the tool->boundary mapping."""

from __future__ import annotations

from asb_target.config_specs import CONFIG_SPEC_NAMES, CONFIG_SPECS
from asb_target.controllables import CONTROLLABLES, opi_tool_observation_ctrl
from asb_target.observables import STATIC_OBSERVABLE_SPECS
from asb_target.security_tags import (
    MEMORY_TAG,
    SCENARIO_TOOL_TAGS,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_OBSERVATION_TAGS,
    TOOLS_TAG,
    USER_TAG,
)
from asb_target.tool_boundary import tool_boundary_tag


def test_controllables_one_per_method_plus_opi_leaves() -> None:
    by_name: dict[str, list] = {}
    for c in CONTROLLABLES:
        by_name.setdefault(c.name, []).append(c)
    assert set(by_name) == {
        "dpi_user_prompt",
        "pot_system_demonstration",
        "mp_retrieved_workflow",
        "opi_tool_observation",
    }
    assert len(by_name["dpi_user_prompt"]) == 1
    assert by_name["dpi_user_prompt"][0].security_domain is USER_TAG
    assert by_name["pot_system_demonstration"][0].security_domain is SYSTEM_PROMPT_TAG
    assert by_name["mp_retrieved_workflow"][0].security_domain is MEMORY_TAG
    # one OPI controllable per scenario sub-boundary
    assert len(by_name["opi_tool_observation"]) == len(SCENARIO_TOOL_TAGS)
    opi_domains = {c.security_domain for c in by_name["opi_tool_observation"]}
    assert opi_domains == set(SCENARIO_TOOL_TAGS.values())


def test_opi_builder_carries_passed_tag() -> None:
    tag = SCENARIO_TOOL_TAGS["financial_analyst"]
    ctrl = opi_tool_observation_ctrl(tag)
    assert ctrl.name == "opi_tool_observation"
    assert ctrl.security_domain is tag


def test_mp_description_has_no_stale_single_run_text() -> None:
    mp = next(c for c in CONTROLLABLES if c.name == "mp_retrieved_workflow")
    assert "single-run" not in mp.description.lower()
    assert "no cross-run" not in mp.description.lower()


def test_config_slots_exactly_four() -> None:
    assert CONFIG_SPEC_NAMES == {"agent_name", "user_prompt", "attacker_tool", "memory_mode"}
    by = {s.name: s for s in CONFIG_SPECS}
    assert by["attacker_tool"].security_domain is SYSTEM_TAG
    assert by["memory_mode"].security_domain is MEMORY_TAG
    assert by["agent_name"].security_domain is SYSTEM_TAG
    assert by["user_prompt"].security_domain is USER_TAG


def test_removed_config_slots_absent() -> None:
    for removed in ("defense_type", "pot_trigger", "llm_model"):
        assert removed not in CONFIG_SPEC_NAMES


def test_static_observables_trimmed() -> None:
    names = {o.name for o in STATIC_OBSERVABLE_SPECS}
    assert names == {"system_prompt", "tool_catalog_listing"}
    assert "model_identity" not in names
    assert not any("attack_reference" in n for n in names)
    by = {o.name: o for o in STATIC_OBSERVABLE_SPECS}
    assert by["system_prompt"].security_domain is SYSTEM_PROMPT_TAG
    assert by["tool_catalog_listing"].security_domain is TOOLS_TAG


def test_tool_boundary_mapping_complete_and_mechanical() -> None:
    for name, tag in TOOL_OBSERVATION_TAGS.items():
        assert tool_boundary_tag(name) is tag
    # unknown / attacker tool -> the tools root (still in scope for any {tools} attacker)
    assert tool_boundary_tag("nonexistent_tool") is TOOLS_TAG
    assert tool_boundary_tag("AggressiveResourceHijack") is TOOLS_TAG


def test_no_judgment_buckets() -> None:
    # mechanical per-tool leaves, no functional grouping tags
    leaf_names = {t.name for t in TOOL_OBSERVATION_TAGS.values()}
    for bucket in ("tools.knowledge", "tools.telemetry", "tools.records", "tools.compute"):
        assert bucket not in leaf_names
    assert TOOLS_TAG.name == "tools"
    assert SYSTEM_TAG.name == "system"
