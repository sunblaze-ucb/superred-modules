"""Tests for the Observable enumeration."""

from __future__ import annotations

import pytest

from agentdojo_target.controllables import READ_CTRLS
from agentdojo_target.observables import (
    COMPOSITE_ENV_SNAPSHOT_OBS,
    MODEL_IDENTITY_OBS,
    STATIC_OBSERVABLE_SPECS,
    SYSTEM_PROMPT_OBS,
    TOOL_CATALOG_LISTING_OBS,
    agent_tool_call_observable,
    agent_tool_response_observable,
    chat_message_observable,
    read_data_field_observable,
    write_call_observable,
)
from agentdojo_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    AGENT_TRACE_TOOL_RESPONSES_TAG,
    MODEL_IDENTITY_TAG,
    PROMPT_READABLE_TAG,
    TOOL_CATALOGUE_READABLE_TAG,
    TOOLS_TAG,
)


def test_static_specs_have_right_tags() -> None:
    assert MODEL_IDENTITY_OBS.security_domain is MODEL_IDENTITY_TAG
    assert SYSTEM_PROMPT_OBS.security_domain is PROMPT_READABLE_TAG
    assert TOOL_CATALOG_LISTING_OBS.security_domain is TOOL_CATALOGUE_READABLE_TAG
    assert COMPOSITE_ENV_SNAPSHOT_OBS.security_domain is TOOLS_TAG


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


def test_write_call_observable_uses_tool_calls_tag() -> None:
    obs = write_call_observable("workspace__send_email")
    assert obs.security_domain is AGENT_TRACE_TOOL_CALLS_TAG
    assert "workspace__send_email" in obs.name


def test_read_data_field_observable_mirrors_controllable_tag() -> None:
    """The mirror observable shares the controllable's 2x2 leaf tag."""
    for prefixed, ctrl in READ_CTRLS.items():
        obs = read_data_field_observable(prefixed)
        assert obs.security_domain is ctrl.security_domain, (
            f"Observable for {prefixed} on {obs.security_domain.name}; "
            f"controllable on {ctrl.security_domain.name}"
        )


def test_read_data_field_observable_unknown_tool_raises() -> None:
    with pytest.raises(KeyError, match="No read Controllable"):
        read_data_field_observable("workspace__send_email")  # write tool
