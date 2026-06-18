"""Unit tests for the target's exposed surface: controllables, config slots,
observables, and the tool->boundary mapping."""

from __future__ import annotations

from asb_target.config_specs import CONFIG_SPEC_NAMES, CONFIG_SPECS
from asb_target.controllables import CONTROLLABLES, opi_tool_observation_ctrl
from asb_target.observables import STATIC_OBSERVABLE_SPECS
from asb_target.security_tags import (
    MEMORY_TAG,
    MODEL_IDENTITY_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_CATALOGUE_ADD_TAG,
    TOOL_CATALOGUE_EDIT_TAG,
    TOOL_CATALOGUE_REMOVE_TAG,
    TOOL_CATALOGUE_TAG,
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
        "tool_catalog_register",
        "tool_catalog_replace",
        "tool_catalog_unregister",
        "tool_catalog_rewrite_doc",
        "opi_tool_observation",
    }
    assert len(by_name["dpi_user_prompt"]) == 1
    assert by_name["dpi_user_prompt"][0].security_domain is USER_TAG
    assert by_name["pot_system_demonstration"][0].security_domain is SYSTEM_PROMPT_TAG
    assert by_name["mp_retrieved_workflow"][0].security_domain is MEMORY_TAG
    # tool-catalogue edit controllables, tagged at the add/edit/remove capability
    assert by_name["tool_catalog_register"][0].security_domain is TOOL_CATALOGUE_ADD_TAG
    assert by_name["tool_catalog_replace"][0].security_domain is TOOL_CATALOGUE_EDIT_TAG
    assert by_name["tool_catalog_rewrite_doc"][0].security_domain is TOOL_CATALOGUE_EDIT_TAG
    assert by_name["tool_catalog_unregister"][0].security_domain is TOOL_CATALOGUE_REMOVE_TAG
    # one OPI controllable per tool leaf (fully per-tool granular)
    assert len(by_name["opi_tool_observation"]) == len(TOOL_OBSERVATION_TAGS) == 20
    opi_domains = {c.security_domain for c in by_name["opi_tool_observation"]}
    assert opi_domains == set(TOOL_OBSERVATION_TAGS.values())


def test_opi_builder_carries_passed_tag() -> None:
    tag = TOOL_OBSERVATION_TAGS["market_data_api"]
    ctrl = opi_tool_observation_ctrl(tag)
    assert ctrl.name == "opi_tool_observation"
    assert ctrl.security_domain is tag


def test_mp_description_has_no_stale_single_run_text() -> None:
    mp = next(c for c in CONTROLLABLES if c.name == "mp_retrieved_workflow")
    assert "single-run" not in mp.description.lower()
    assert "no cross-run" not in mp.description.lower()


def test_config_slots() -> None:
    assert CONFIG_SPEC_NAMES == {
        "agent_name",
        "user_prompt",
        "attacker_tool",
        "memory_mode",
        "attacker_tool_forcing",
    }
    by = {s.name: s for s in CONFIG_SPECS}
    assert by["attacker_tool"].security_domain is SYSTEM_TAG
    assert by["memory_mode"].security_domain is MEMORY_TAG
    assert by["agent_name"].security_domain is SYSTEM_TAG
    assert by["user_prompt"].security_domain is USER_TAG
    assert by["attacker_tool_forcing"].security_domain is SYSTEM_TAG


def test_removed_config_slots_absent() -> None:
    for removed in ("defense_type", "pot_trigger", "llm_model"):
        assert removed not in CONFIG_SPEC_NAMES


def test_static_observables() -> None:
    names = {o.name for o in STATIC_OBSERVABLE_SPECS}
    assert names == {"system_prompt", "model_identity", "tool_catalog_listing"}
    assert not any("attack_reference" in n for n in names)
    by = {o.name: o for o in STATIC_OBSERVABLE_SPECS}
    assert by["system_prompt"].security_domain is SYSTEM_PROMPT_TAG
    assert by["model_identity"].security_domain is MODEL_IDENTITY_TAG
    # The catalogue LISTING is tagged at the registry boundary (tool_catalogue),
    # NOT the tools tree, so granting catalogue-read does not grant reading any
    # tool's returned observations.
    assert by["tool_catalog_listing"].security_domain is TOOL_CATALOGUE_TAG


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


def test_shadow_call_ctrl_tagged_at_capability_not_tool_leaf() -> None:
    # The requested semantics: an attacker-managed tool's per-call event is
    # tagged at the add/edit CAPABILITY, so the editing scope grants control of
    # the call even when the tool's own tools.* leaf is out of scope.
    from superred.core.types.security_domain import scope_includes

    from asb_target.controllables import tool_catalogue_call_ctrl

    add_ctrl = tool_catalogue_call_ctrl(TOOL_CATALOGUE_ADD_TAG, "evil_exfil")
    edit_ctrl = tool_catalogue_call_ctrl(TOOL_CATALOGUE_EDIT_TAG, "sys_monitor")
    assert add_ctrl.security_domain is TOOL_CATALOGUE_ADD_TAG
    assert edit_ctrl.security_domain is TOOL_CATALOGUE_EDIT_TAG
    assert add_ctrl.name == "tool_call:evil_exfil"

    edit_scope = frozenset({TOOL_CATALOGUE_EDIT_TAG})
    # edit-scope grants the per-call shadow event ...
    assert scope_includes(edit_scope, edit_ctrl.security_domain)
    # ... even though the tool's own tools.* leaf is NOT in that scope
    assert not scope_includes(edit_scope, TOOL_OBSERVATION_TAGS["sys_monitor"])
