"""Offline surface tests for the dtap_scaffold pure layer (no Docker/LLM)."""

from __future__ import annotations

import pytest
from superred.core.types.controllable import Controllable
from superred.core.types.observable import Observable
from superred.core.types.security_domain import (
    SecurityDomain,
    SecurityDomainTag,
    scope_includes,
)
from superred.core.types.state import ConfigSpec, QuerySpec

import dtap_scaffold as S  # noqa: N812


def test_default_domain_roots():
    assert isinstance(S.DOMAIN, SecurityDomain)
    roots = {t.name for t in S.DOMAIN.roots}
    assert roots == {"system", "user", "tools", "environment", "host"}


def test_subsumption_hierarchy():
    # parent includes its descendants (identity-based)
    assert S.SYSTEM_TAG.includes(S.SYSTEM_PROMPT_TAG)
    assert S.SYSTEM_TAG.includes(S.TOOL_CATALOGUE_EDIT_TAG)
    assert S.TOOL_CATALOGUE_TAG.includes(S.TOOL_CATALOGUE_EDIT_TAG)
    assert S.AGENT_TRACE_TAG.includes(S.AGENT_TRACE_TOOL_CALLS_TAG)
    # host root subsumes both host capabilities; the two are independent of each other
    assert S.HOST_TAG.includes(S.HOST_FILESYSTEM_TAG)
    assert S.HOST_TAG.includes(S.HOST_CODE_EXECUTION_TAG)
    assert not S.HOST_FILESYSTEM_TAG.includes(S.HOST_CODE_EXECUTION_TAG)
    # not the other way round
    assert not S.SYSTEM_PROMPT_TAG.includes(S.SYSTEM_TAG)
    assert not S.USER_TAG.includes(S.SYSTEM_TAG)
    assert not S.HOST_FILESYSTEM_TAG.includes(S.HOST_TAG)


def test_fixed_controllables():
    assert [c.name for c in S.FIXED_CONTROLLABLES] == [
        "user_prompt",
        "system_prompt",
        "skill",
        "tool_description_override",
        "tool_description_suffix",
        "filesystem",
        "code_execution",
    ]
    assert all(isinstance(c, Controllable) for c in S.FIXED_CONTROLLABLES)
    assert S.USER_PROMPT_CTRL.security_domain is S.USER_TAG
    assert S.SYSTEM_PROMPT_CTRL.security_domain is S.SYSTEM_PROMPT_TAG
    assert S.SKILL_CTRL.security_domain is S.SKILL_TAG
    assert S.SKILL_CTRL.value_type == "json"
    assert S.TOOL_DESCRIPTION_OVERRIDE_CTRL.security_domain is S.TOOL_CATALOGUE_EDIT_TAG
    assert S.TOOL_DESCRIPTION_SUFFIX_CTRL.security_domain is S.TOOL_CATALOGUE_EDIT_TAG
    # the host trust boundary: filesystem (PreCall) + code_execution (PostCall loop)
    assert S.FILESYSTEM_CTRL.security_domain is S.HOST_FILESYSTEM_TAG
    assert S.FILESYSTEM_CTRL.value_type == "json"
    assert S.CODE_EXECUTION_CTRL.security_domain is S.HOST_CODE_EXECUTION_TAG
    assert S.CODE_EXECUTION_CTRL.value_type == "text"


def test_env_controllable_builders():
    tag = S.tools_server_tag("salesforce")
    # root/fallback surface (node_key="") keeps the bare env_tool:<server> name
    ctrl = S.env_tool_output_controllable("salesforce", "", tag)
    assert ctrl.name == "env_tool:salesforce"
    assert ctrl.security_domain is tag
    assert ctrl.value_type == "text"
    # a named node yields env_tool:<server>.<node> scoped to the node tag
    node = S.build_server_tree("salesforce").nodes["base"]
    node_ctrl = S.env_tool_output_controllable("salesforce", "base", node)
    assert node_ctrl.name == "env_tool:salesforce.base"
    assert node_ctrl.security_domain is node

    etag = S.env_server_tag("gmail-injection")
    inj = S.env_inject_controllable("gmail-injection", etag)
    assert inj.name == "env_inject:gmail-injection"
    assert inj.security_domain is etag
    assert inj.value_type == "json"


def test_static_observables():
    assert all(isinstance(o, Observable) for o in S.STATIC_OBSERVABLE_SPECS)
    names = {o.name for o in S.STATIC_OBSERVABLE_SPECS}
    assert names == {
        "model_identity",
        "detailed_system_specification",
        "tool_catalog_listing",
        "active_environments",
        "max_turns",
    }


def test_dynamic_observable_builders():
    msg = S.agent_message_observable(3)
    assert msg.name == "agent_trace_message_0003"
    assert msg.security_domain is S.AGENT_TRACE_MESSAGES_TAG
    tool = S.native_tool_observable(7)
    assert tool.name == "native_tool_call_0007"
    assert tool.security_domain is S.AGENT_TRACE_TOOL_CALLS_TAG


def test_specs():
    cfg_names = {c.name for c in S.CONFIG_SPECS}
    assert {
        "active_mcp_servers",
        "env_injection_config",
        "system_prompt",
        "user_prompt",
        "task_dir",
        "native_tools_policy",
    } <= cfg_names
    assert all(isinstance(c, ConfigSpec) for c in S.CONFIG_SPECS)
    q_names = {q.name for q in S.QUERY_SPECS}
    assert q_names == {
        "final_response",
        "agent_responses",
        "trajectory_json",
        "env_ports",
        "task_dir",
    }
    assert all(isinstance(q, QuerySpec) for q in S.QUERY_SPECS)


def test_text_only_domains():
    assert len(S.TEXT_ONLY_DOMAINS) == 11
    assert S.is_text_only_domain("travel")
    assert not S.is_text_only_domain("browser")
    assert S.require_text_only_domain("os-filesystem") == "os-filesystem"
    with pytest.raises(ValueError):
        S.require_text_only_domain("browser")
    with pytest.raises(ValueError):
        S.require_text_only_domain("nonsense")


def test_build_domain_with_active_servers_and_scope_identity():
    # The leaf builders are cached, so every caller shares the ONE instance.
    tools_tag = S.tools_server_tag("salesforce")
    env_tag = S.env_server_tag("salesforce-injection")
    domain = S.build_domain([tools_tag], [env_tag])
    assert isinstance(domain, SecurityDomain)
    by_name = {t.name for t in domain._tags.values()}  # noqa: SLF001 - test introspection
    assert "tools.salesforce" in by_name
    assert "environment.salesforce-injection" in by_name

    # A scope holding a dynamic root covers its (cached) leaf by identity.
    assert scope_includes(frozenset({S.TOOLS_TAG}), tools_tag)
    assert scope_includes(frozenset({S.ENVIRONMENT_TAG}), env_tag)
    assert scope_includes(frozenset({tools_tag}), tools_tag)
    # Caching centralizes identity: a repeat call returns the SAME instance...
    assert S.tools_server_tag("salesforce") is tools_tag
    assert S.env_server_tag("salesforce-injection") is env_tag
    # ...but a hand-built, equal-but-not-identical tag is NOT covered -- why the
    # identity (`is`, not `==`) semantics mean tags must come from the builders.
    hand_built = SecurityDomainTag("tools.salesforce", parent=S.TOOLS_TAG)
    assert not scope_includes(frozenset({tools_tag}), hand_built)
