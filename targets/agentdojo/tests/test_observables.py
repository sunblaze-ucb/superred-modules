"""Tests for the Observable enumeration."""

from __future__ import annotations

from agentdojo_target.observables import (
    COMPOSITE_ENV_SNAPSHOT_OBS,
    MODEL_IDENTITY_OBS,
    STATIC_OBSERVABLE_SPECS,
    TOOL_CATALOG_LISTING_OBS,
    agent_tool_call_observable,
    agent_tool_response_observable,
    chat_message_observable,
)
from agentdojo_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    AGENT_TRACE_TOOL_RESPONSES_TAG,
    MODEL_IDENTITY_TAG,
    TOOL_CATALOGUE_TAG,
    TOOLS_TAG,
)


def test_static_specs_have_right_tags() -> None:
    assert MODEL_IDENTITY_OBS.security_domain is MODEL_IDENTITY_TAG
    # The catalogue listing carries the tool_catalogue tag itself; read-only
    # access is granted at the Controller by listing the tag under read_only.
    assert TOOL_CATALOG_LISTING_OBS.security_domain is TOOL_CATALOGUE_TAG
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
