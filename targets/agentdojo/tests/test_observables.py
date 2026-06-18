"""Tests for the Observable enumeration."""

from __future__ import annotations

from agentdojo_target.observables import (
    DETAILED_SYSTEM_SPECIFICATION_OBS,
    MODEL_IDENTITY_OBS,
    STATIC_OBSERVABLE_SPECS,
    TOOL_CATALOG_LISTING_OBS,
    catalog_edit_outcome_observable,
    chat_message_observable,
    discarded_action_observable,
    tool_menu_rebuild_observable,
    write_observation_observable,
)
from agentdojo_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TAG,
    BANKING_BANK_ACCOUNT_TAG,
    DETAILED_SYSTEM_SPECIFICATION_TAG,
    MODEL_IDENTITY_TAG,
    TOOL_CATALOGUE_ADD_TAG,
    TOOL_CATALOGUE_EDIT_TAG,
    TOOL_CATALOGUE_TAG,
)


def test_static_specs_have_right_tags() -> None:
    assert MODEL_IDENTITY_OBS.security_domain is MODEL_IDENTITY_TAG
    # The catalogue listing carries the tool_catalogue tag itself; read-only
    # access is granted at the Controller by listing the tag under read_only.
    assert TOOL_CATALOG_LISTING_OBS.security_domain is TOOL_CATALOGUE_TAG
    # The leaked system-spec brief is a read-only system child, its own tag.
    assert DETAILED_SYSTEM_SPECIFICATION_OBS.security_domain is DETAILED_SYSTEM_SPECIFICATION_TAG
    # The static specs are now exactly these three observables, each at its tag.
    assert set(STATIC_OBSERVABLE_SPECS) == {
        MODEL_IDENTITY_OBS,
        DETAILED_SYSTEM_SPECIFICATION_OBS,
        TOOL_CATALOG_LISTING_OBS,
    }
    assert {(o.name, o.security_domain) for o in STATIC_OBSERVABLE_SPECS} == {
        ("model_identity", MODEL_IDENTITY_TAG),
        ("detailed_system_specification", DETAILED_SYSTEM_SPECIFICATION_TAG),
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


def test_write_observation_observable_uses_store_tag() -> None:
    obs = write_observation_observable(0, BANKING_BANK_ACCOUNT_TAG)
    assert obs.name == "write_call_0000"
    assert obs.security_domain is BANKING_BANK_ACCOUNT_TAG


def test_tool_menu_rebuild_observable_uses_trace_tag() -> None:
    obs = tool_menu_rebuild_observable(0)
    assert obs.name == "tool_menu_rebuild_0000"
    assert obs.security_domain is AGENT_TRACE_TAG


def test_discarded_action_observable_uses_messages_tag() -> None:
    """A discarded malformed action belongs to the non-tool agent-trace
    message stream (it never becomes a real tool interaction)."""
    obs = discarded_action_observable(1)
    assert obs.name == "discarded_action_0001"
    assert obs.security_domain is AGENT_TRACE_MESSAGES_TAG


def test_catalog_edit_outcome_observable_uses_passed_tag() -> None:
    register = catalog_edit_outcome_observable("register", TOOL_CATALOGUE_ADD_TAG)
    assert register.name == "catalog_edit_register_outcome"
    assert register.security_domain is TOOL_CATALOGUE_ADD_TAG

    replace = catalog_edit_outcome_observable("replace", TOOL_CATALOGUE_EDIT_TAG)
    assert replace.name == "catalog_edit_replace_outcome"
    assert replace.security_domain is TOOL_CATALOGUE_EDIT_TAG
