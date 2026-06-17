"""Tests for the Observable enumeration."""

from __future__ import annotations

from agentdojo_target.observables import (
    MODEL_IDENTITY_OBS,
    STATIC_OBSERVABLE_SPECS,
    TOOL_CATALOG_LISTING_OBS,
    agent_tool_call_observable,
    agent_tool_response_observable,
    catalog_edit_outcome_observable,
    chat_message_observable,
    discarded_action_observable,
    tool_menu_rebuild_observable,
    write_observation_observable,
)
from agentdojo_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    AGENT_TRACE_TOOL_RESPONSES_TAG,
    BANKING_BANK_ACCOUNT_TAG,
    MODEL_IDENTITY_TAG,
    TOOL_CATALOGUE_ADDABLE_TAG,
    TOOL_CATALOGUE_TAG,
)


def test_static_specs_have_right_tags() -> None:
    assert MODEL_IDENTITY_OBS.security_domain is MODEL_IDENTITY_TAG
    # The catalogue listing carries the tool_catalogue tag itself; read-only
    # access is granted at the Controller by listing the tag under read_only.
    assert TOOL_CATALOG_LISTING_OBS.security_domain is TOOL_CATALOGUE_TAG
    # The static specs are now exactly these two observables, each at its tag.
    assert set(STATIC_OBSERVABLE_SPECS) == {
        MODEL_IDENTITY_OBS,
        TOOL_CATALOG_LISTING_OBS,
    }
    assert {(o.name, o.security_domain) for o in STATIC_OBSERVABLE_SPECS} == {
        ("model_identity", MODEL_IDENTITY_TAG),
        ("tool_catalog_listing", TOOL_CATALOGUE_TAG),
    }
    # No static observable carries the (now-removed) TOOLS_TAG.
    assert all(o.security_domain.name != "tools" for o in STATIC_OBSERVABLE_SPECS)


def test_static_specs_no_duplicates() -> None:
    names = [o.name for o in STATIC_OBSERVABLE_SPECS]
    assert len(names) == len(set(names))


def test_chat_message_observable_uses_messages_tag() -> None:
    obs = chat_message_observable(0)
    assert obs.security_domain is AGENT_TRACE_MESSAGES_TAG
    assert "0000" in obs.name


def test_agent_tool_call_observable_uses_tool_calls_tag() -> None:
    obs = agent_tool_call_observable(5)
    assert obs.security_domain is AGENT_TRACE_TOOL_CALLS_TAG


def test_agent_tool_response_observable_uses_tool_responses_tag() -> None:
    obs = agent_tool_response_observable(2)
    assert obs.security_domain is AGENT_TRACE_TOOL_RESPONSES_TAG


def test_write_observation_observable_uses_store_tag() -> None:
    obs = write_observation_observable(0, BANKING_BANK_ACCOUNT_TAG)
    assert obs.name == "write_call_0000"
    assert obs.security_domain is BANKING_BANK_ACCOUNT_TAG


def test_tool_menu_rebuild_observable_uses_trace_tag() -> None:
    obs = tool_menu_rebuild_observable(0)
    assert obs.name == "tool_menu_rebuild_0000"
    assert obs.security_domain is AGENT_TRACE_TAG


def test_discarded_action_observable_uses_tool_calls_tag() -> None:
    obs = discarded_action_observable(1)
    assert obs.name == "discarded_action_0001"
    assert obs.security_domain is AGENT_TRACE_TOOL_CALLS_TAG


def test_catalog_edit_outcome_observable_uses_passed_tag() -> None:
    register = catalog_edit_outcome_observable("register", TOOL_CATALOGUE_ADDABLE_TAG)
    assert register.name == "catalog_edit_register_outcome"
    assert register.security_domain is TOOL_CATALOGUE_ADDABLE_TAG

    replace = catalog_edit_outcome_observable("replace", TOOL_CATALOGUE_TAG)
    assert replace.name == "catalog_edit_replace_outcome"
    assert replace.security_domain is TOOL_CATALOGUE_TAG
